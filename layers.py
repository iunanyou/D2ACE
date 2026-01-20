import torch
import torch.nn as nn
import math
from util import *
import torch.nn.functional as F
class CondTDist(nn.Module):
    '''
    A distribution transformer implemented by stacking conditional normalizing flows.
    '''
    def __init__(self, base_dist, conditional_transforms: list):
        super(CondTDist, self).__init__()
        self.base_dist = base_dist
        self.conditional_transforms = nn.ModuleList(conditional_transforms)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self.base_dist, 'reset_parameters'):
            self.base_dist.reset_parameters()
        for t in self.conditional_transforms:
            t.reset_parameters()
            
    def forward(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Transform x into z which follows a kwown base distribution.

        Parameters
        ----------
        x : Tensor [B x * x D]
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.
            
        Returns
        -------
        Tensor [B x * x D]
        '''
        for t in self.conditional_transforms:
            x = t(x, cond_var)
        
        return x
    
    def log_prob(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Compute the log probabilities at x.

        Parameters
        ----------
        x : Tensor [B x * x D]
            Point at which pdf is to be evaluated.
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.
            
        Returns
        -------
        Tensor [B x *]
            Log probabilities at x.
        '''
        log_dets = 0.0
        for t in self.conditional_transforms:
            x = t(x, cond_var)
            log_dets = log_dets + t.log_det()
        
        log_prob_x = self.base_dist.log_prob(x).sum(-1) + log_dets
        
        return log_prob_x
    
    def log_det(self) -> Tensor:
        '''
        Compute the log determinant of the Jacobian.
        '''
        log_dets = 0.0
        for t in self.conditional_transforms:
            log_dets = log_dets + t.log_det()
            
        return log_dets
    
    def sample(self, size: list=[1]) -> Tensor:
        '''
        Sample from the learned distribution.
        '''
        raise NotImplementedError
def get_activation(nonlinearity, param):
    if nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity == 'leaky_relu':
        return nn.LeakyReLU(param, inplace=True)
    elif nonlinearity == 'elu':
        return nn.ELU(param, inplace=True)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
##MLP##
class MLP(nn.Module):
    def __init__(self,input_size,out_size,hidden_list=[],batchNorm=True,drop_ratio=0.2,
                    nonlinearity='leaky_relu',negative_slope=0.1,with_output_nonlineartity=True):
        super(MLP,self).__init__()
        self.fcs=nn.ModuleList()
        self.input_size=input_size
        self.out_size=out_size
        self.nonlinearity=nonlinearity
        self.negative_slope=negative_slope
        if hidden_list:
            in_dims=[input_size]+hidden_list
            out_dims=hidden_list+[out_size]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i],out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_list):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                            self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        #报错
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                    if drop_ratio:
                        self.fcs.append(nn.Dropout(drop_ratio))
        else:
            self.fcs.append(nn.Linear(input_size,out_size))
            if with_output_nonlineartity:
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                    
        self.reset_parameters()
    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                            nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()
    def forward(self,x):
        for fc in self.fcs:
            x = fc(x)
        return x

##GINlayer
class GINLayer(nn.Module):
    def __init__(self,mlp, eps=0.0,residual=True,train_eps=True):
        super(GINLayer,self).__init__()
        self.mlp=mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.residual=residual
        self.reset_parameters()
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self,input,adj):
        '''
        input:[node_size,emb_size]
        adj:[node_size,node_size]
        '''
        res=input
        neighs=torch.matmul(adj,res)
        res=(1+self.eps)*res+neighs

        res=self.mlp(res)
        if self.residual:
            res=res+input
        return res         

#GIN
class GIN(nn.Module):
    def __init__(self, num_layers, input_size, out_size, hidden_list=[],
                 eps=0.0,drop_ratio=0.2, train_eps=True, residual=True, batchNorm=True,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(GIN,self).__init__()
        self.GINLayers=nn.ModuleList()
        if input_size!=out_size:
            first_layer_res=False
        else:
            first_layer_res=True
        self.GINLayers.append(GINLayer(MLP(input_size,out_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope),eps,first_layer_res))
        for i in range(num_layers-1):
            self.GINLayers.append(GINLayer(MLP(out_size,out_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope),eps,residual))
            
        self.reset_parameters()
    
    def reset_parameters(self):
        for l in self.GINLayers:
            l.reset_parameters()
    def forward(self,input,adj):
        for gin in self.GINLayers:
            input=gin(input,adj)
        return input



#FD
class FDModel(nn.Module):
    def __init__(self, input_x_size,input_y_size, hidden_size, out_size,
                 in_layers1=1, out_layers=1, batchNorm=False,
                 nonlinearity='leaky_relu',drop_ratio=0.2, negative_slope=0.1):
        super(FDModel, self).__init__()
        hidden_list=[hidden_size]*(in_layers1-1)
        self.out_x=MLP(input_x_size,hidden_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope)

        self.out_y=nn.Linear(input_y_size,hidden_size)

        hidden_list=[hidden_size]*(out_layers-1)
        self.out=MLP(hidden_size,out_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.out_x.reset_parameters()
        nn.init.kaiming_uniform_(self.out_y.weight, nonlinearity='sigmoid')
        nn.init.constant_(self.out_y.bias, 0.0)
        self.out.reset_parameters()
    
    def forward(self, x, y):
        x=self.out_x(x)#[b1,hidden]
        y=self.out_y(y)#[b2,hidden]
        out=x.unsqueeze(dim=1)*y.unsqueeze(dim=0)#[b1,b2,h]

        out=self.out(out)#[b1,b2,out_size]
        return out



class Conditioner(nn.Module):
    '''
    An autoregressive conditioner.
    params = c(x_{1:i-1},y)
    '''
    def __init__(self, in_dim: int, cond_dim: int, out_param_dim: int,
                 h_dim: list=[], input_order: str='same', mode: str='sequential',
                 nonlinearity: str='elu', act_param: float=1.0):
        '''
        Parameters
        ----------
        in_dim : int
            Dimension of the input.
        cond_dim : int
            Dimension of the conditional variable y.
        out_param_dim : int
            Dimension of the parameters.
        h_dim : list, optional
            List with number of hidden units for each hidden layer.
            The default is [].
        input_order : str, optional
            Strategy for assigning degrees to input units, which can be 'random',
            'same' or 'inverse'.
            The default is 'same'.
        mode : str, optional
            Strategy for assigning degrees to hidden units, which can be 'random'
            or 'sequential'.
            The default is 'sequential'.
        nonlinearity : str, optional
            Nonlinearity used in neural networks.
            The default is 'elu'.
        act_param : float, optional
            Parameter for some nonlinearity.
            The default is 1.0.
        '''
        super(Conditioner, self).__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.out_param_dim = out_param_dim
        self.out_dim = in_dim * out_param_dim
        self.h_dim = h_dim
        self.input_order = input_order
        self.mode = mode
        
        # Assign degrees to each unit
        degrees = self._assign_degrees()
        # Create masks
        masks = self._create_masks(degrees)
        
        # Create models
        activation = get_activation(nonlinearity, act_param)
        self.mlp = CondMaskedMLP(in_dim, cond_dim, self.out_dim, h_dim, masks, activation)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        degrees = self._assign_degrees()
        masks = self._create_masks(degrees)
        self.mlp.reset_parameters(masks)
            
    def forward(self, input: Tensor, cond_var: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.mlp(input, cond_var)
        output = output.view(*output.size()[:-1], self.out_param_dim, self.in_dim)
        with torch.no_grad():
            dim = output.dim()
            perm = list(range(dim-2)) + [dim-1, dim-2]
        
        return output.permute(perm)
    
    def _assign_degrees(self) -> list:
        '''
        Assign a degree for each hidden and input unit. A unit with degree d can only receive input from units with
        degree less than d.
        '''
        degrees = []
        if self.input_order == 'random':
            degrees_0 = torch.randperm(self.in_dim) + 1
        elif self.input_order == 'same':
            degrees_0 = torch.arange(1, self.in_dim+1)
        elif self.input_order == 'inverse':
            degrees_0 = torch.arange(self.in_dim, 0, -1)
        else:
            raise ValueError("Unsupported input_order {}".format(self.input_order))
        degrees.append(degrees_0)
        
        if self.mode == 'random':
            for N in self.h_dim:
                min_prev_degree = torch.min(degrees[-1])
                degrees_l = torch.randint(min_prev_degree, max(self.in_dim,2), N)
                degrees.append(degrees_l)
        elif self.mode == 'sequential':
            for N in self.h_dim:
                degrees_l = torch.arange(N) % max(self.in_dim-1, 1) + 1
                degrees.append(degrees_l)
        else:
            raise ValueError("Unsupported mode {}".format(self.mode))
        
        return degrees
    
    def _create_masks(self, degrees: list) -> Tuple[list, Tensor]:
        '''
        Create binary masks that make the connectivity autoregressive.
        '''
        masks = []
        for d0, d1 in zip(degrees[:-1], degrees[1:]):
            masks.append(d0.unsqueeze(0) <= d1.unsqueeze(1))
            
        output_mask = degrees[-1].unsqueeze(0) < degrees[0].unsqueeze(1)
        masks.append(output_mask.repeat(self.out_param_dim, 1))
        
        return masks
        
class CondMaskedMLP(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, out_dim: int, h_dim: list,
                 masks: Tensor, activation):
        '''
        Parameters
        ----------
        in_dim : int
            Dimension of the input.
        cond_dim : int
            Dimension of the conditional variable y.
        out_dim : int
            Dimension of the output.
        h_dim : list
            List with number of hidden units for each hidden layer.
        masks : Tensor
            Masks for weights.
        activation
            Nonlinearity function used in neural networks.
        '''
        super(CondMaskedMLP, self).__init__()
        self.activation = activation
        
        self.fcs = nn.ModuleList()
        self.masks = nn.ParameterList()
        
        if h_dim:
            in_dims = [in_dim] + h_dim[:-1]
            out_dims = h_dim
            next_dim = h_dim[-1]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                self.masks.append(nn.Parameter(masks[i], requires_grad=False))
            self.fc_y = nn.Linear(cond_dim, h_dim[0])
        else:
            next_dim = in_dims
            self.fc_y = nn.Linear(cond_dim, out_dim)
        self.fcs.append(nn.Linear(next_dim, out_dim))
        self.masks.append(nn.Parameter(masks[-1], requires_grad=False))
                
        self.reset_parameters()
        
    def reset_parameters(self, masks: Tensor=None):
        for l in self.fcs:
            l.reset_parameters()
        
        if masks:
            for i, m in enumerate(masks):
                self.masks[i].data = m
        nn.init.kaiming_uniform_(self.fc_y.weight, a=math.sqrt(5))
        nn.init.constant_(self.fc_y.bias, 0.0)
    
    def forward(self, input: Tensor, cond_var: Tensor):
        if len(self.fcs) == 1:
            return masked_linear(self.fcs[0], self.masks[0], input) + self.fc_y(cond_var)
        
        input = self.activation(masked_linear(self.fcs[0], self.masks[0], input)
                                + self.fc_y(cond_var))
        for i in range(1, len(self.fcs)-1):
            input = self.activation(masked_linear(self.fcs[i], self.masks[i], input))
        input = masked_linear(self.fcs[-1], self.masks[-1], input)
            
        return input

def masked_linear(fc, mask: Tensor, input: Tensor) -> Tensor:
    '''
    A Linear layer with mask.
    '''
    return F.linear(input, fc.weight * mask, fc.bias)
class CondAF(nn.Module):
    '''
    Affine Flow for modeling conditional probability p(x|y).
    '''
    def __init__(self, in_features: int, cond_features: int, nonlinearity_a: str='softpuls',
                 identity_init: bool=False):
        '''
        Parameters
        ----------
        in_features : int
            Dimension of the input.
        cond_features : int
            Dimension of the conditional variable y.
        nonlinearity_a : str, optional
            Nonlinearity for output a, which guarantees a are all positive.
            The default is 'exp'.
        identity_init : bool, optional
            Whether to initialize the flow as an identity flow.
            The default is False.
        '''
        super(CondAF, self).__init__()
        self.in_features = in_features
        self.identity_init = identity_init
        
        # Create affine parameters
        self.affine_a = ResLinear(cond_features, in_features)
        self.affine_b = ResLinear(cond_features, in_features)
        
        if nonlinearity_a == 'exp':
            self.nonlinearity_a = 'torch.exp'
        elif nonlinearity_a == 'softpuls':
            self.nonlinearity_a = 'F.softplus'
        else:
            raise ValueError("Unsupported nonlinearity_a {}".format(nonlinearity_a))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.affine_a.reset_parameters()
        self.affine_b.reset_parameters()
        if self.identity_init:
            # let b = 0
            self.affine_b.l2.weight.data.uniform_(-0.001, 0.001)
            self.affine_b.l3.weight.data.uniform_(-0.001, 0.001)
            self.affine_b.l2.bias.data.fill_(0.0)
            self.affine_b.l3.bias.data.fill_(0.0)
            # let a = 1
            self.affine_a.l2.weight.data.uniform_(-0.001, 0.001)
            self.affine_a.l3.weight.data.uniform_(-0.001, 0.001)
            if self.nonlinearity_a == 'torch.exp':
                inv = 0.0
            elif self.nonlinearity_a == 'F.softplus':
                inv = math.log(math.exp(1) - 1) * 0.5
            self.affine_a.l2.bias.data.fill_(inv)
            self.affine_a.l3.bias.data.fill_(inv)
            
    def forward(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Transform x into z which follows a standard norm distribution.

        Parameters
        ----------
        x : Tensor [B x * x D]
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.

        Returns
        -------
        Tensor [B x * x D]
        '''
        self.a = eval(self.nonlinearity_a)(self.affine_a(cond_var))
        b = self.affine_b(cond_var)
        z = x * self.a + b
        
        return z
        
    def log_prob(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Compute the log probabilities at x.

        Parameters
        ----------
        x : Tensor [B x * x D]
            Point at which pdf is to be evaluated.
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.

        Returns
        -------
        Tensor [B x *]
            Log probabilities at x.
        '''
        z = self.forward(x, cond_var)
        
        log_prob_z = -0.5 * math.log(2 * math.pi) -0.5 * z**2
        log_prob_x = log_prob_z.sum(-1) + self.log_det()

        return log_prob_x
    
    def log_det(self) -> Tensor:
        '''
        Compute the log determinant of the Jacobian.
        '''
        return torch.log(self.a).sum(-1)
    
    def sample(self, size: list=[1]) -> Tensor:
        '''
        Sample from the learned distribution.
        '''
        raise NotImplementedError
        
class ResLinear(nn.Module):
    '''
    Linear layer with residual connection.
    z = linear2(act(linear1(x))) + linear3(x)
    '''
    def __init__(self, in_features: int, out_features: int,
                 nonlinearity: str='relu', act_param: float=0.1, 
                 has_res_layer: bool=True):
        super(ResLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_res_layer = has_res_layer
        
        self.l1 = nn.Linear(in_features, out_features)
        self.l2 = nn.Linear(out_features, out_features)
        if self.has_res_layer:
            self.l3 = nn.Linear(in_features, out_features)
        self.activation = get_activation(nonlinearity, act_param)
        
        self.reset_parameters()

    def reset_parameters(self, mask: Tensor=None):
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        if self.has_res_layer:
            self.l3.reset_parameters()
        
    def forward(self, input: Tensor):
        z = self.l2(self.activation(self.l1(input)))
        if self.has_res_layer:
            z = z + self.l3(input)
        else:
            z = z + input
        
        return z
class CondNAF(nn.Module):
    '''
    Neural Autoregressive Flow-DSF [1] for modeling conditional probability p(x|y).
    
    [1] Huang, C.; Krueger, D.; Lacoste, A.; and Courville, A. C. 2018. Neural autoregressive flows.
    In Proceedings of the 35th International Conference on Machine Learning, 2083–2092. Stockholm, Sweden.
    '''
    def __init__(self, in_features: int, cond_features: int, hidden_features: list=[],
                 input_order: str='same', mode: str='sequential',
                 nonlinearity: str='elu', act_param: float=1.0,
                 ds_dim: int=16, num_ds_layer: int=1, identity_init: bool=False,
                 eps: float=1e-6):
        '''
        Parameters
        ----------
        in_features : int
            Dimension of the input.
        cond_features : int
            Dimension of the conditional variable y.
        hidden_features : list, optional
            List with number of hidden units for each hidden layer.
            The default is [].
        input_order : str, optional
            Strategy for assigning degrees to input units, which can be 'random',
            'same' or 'inverse'.
            The default is 'same'.
        mode : str, optional
            Strategy for assigning degrees to hidden units, which can be 'random'
            or 'sequential'.
            The default is 'sequential'.
        nonlinearity : str, optional
            Nonlinearity used in neural networks.
            The default is 'elu'.
        act_param : float, optional
            Parameter for some nonlinearity.
            The default is 1.0.
        ds_dim : int, optional
            Number of hidden units for the sigmoidal neural network.
            The default is 16.
        num_ds_layer : int, optional
            Number of the sigmoidal neural network.
            The default is 1.
        identity_init : bool, optional
            Whether to initialize the flow as an identity flow.
            The default is False.
        eps : float, optional
            A small constant for numerical stabilization.
            The default is 1e-6.
        '''
        super(CondNAF, self).__init__()
        self.in_features = in_features
        self.ds_dim = ds_dim
        self.num_ds_layer = num_ds_layer
        self.identity_init = identity_init
        self.eps = eps
        
        # Create conditioner
        out_dim1 = 3 * (hidden_features[-1] // in_features) * num_ds_layer
        out_dim2 = 3 * ds_dim * num_ds_layer
        self.conditioner = Conditioner(in_features, cond_features, out_dim1,
                                       hidden_features, input_order, mode,
                                       nonlinearity, act_param)
        self.out_to_dsparams = nn.Linear(out_dim1, out_dim2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conditioner.reset_parameters()
        if self.identity_init:
           self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
           self.out_to_dsparams.bias.data.fill_(0.0)
           # bias for a
           inv = math.log(math.exp(1) - 1)
           nparams = 3 * self.ds_dim
           for i in range(self.num_ds_layer):
               start = i * nparams
               self.out_to_dsparams.bias.data[start:start+self.ds_dim].fill_(inv)
        else:
            self.out_to_dsparams.reset_parameters()
            
    def forward(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Transform x into z which follows a standard norm distribution.

        Parameters
        ----------
        x : Tensor [B x * x D]
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.

        Returns
        -------
        Tensor [B x * x D]
        '''
        params = self.conditioner(x, cond_var) # [B x * x D x out_dim1], note that the params is output with no nonlinearity
        params = self.out_to_dsparams(params) # [B x * x D x out_dim2], (a, b, w)
        
        start = 0
        self.logdet = 0.0
        for i in range(self.num_ds_layer):
            a = F.softplus(params[..., start:start+self.ds_dim]) # [B x * x D x ds_dim]
            start += self.ds_dim
            b = params[..., start:start+self.ds_dim] # [B x * x D x ds_dim]
            start += self.ds_dim
            w_ = params[..., start:start+self.ds_dim] # [B x * x D x ds_dim]
            w = torch.softmax(w_, dim=-1)
            start += self.ds_dim
            
            pre_sigm = a * x.unsqueeze(-1) + b # [B x * x D x ds_dim]
            x_pre = torch.sum(w * torch.sigmoid(pre_sigm), dim=-1) # [B x * x D]
            x_pre_clipped = x_pre * (1-self.eps) + self.eps * 0.5
            x = torch.log(x_pre_clipped / (1 - x_pre_clipped)) # [B x * x D]
            
            logdet = F.log_softmax(w_, dim=-1) + F.logsigmoid(pre_sigm) + \
                      F.logsigmoid(-pre_sigm) + torch.log(a)
            logdet = torch.logsumexp(logdet, dim=-1) + math.log(1-self.eps) - \
                      torch.log(x_pre_clipped) - torch.log(1-x_pre_clipped)
            self.logdet = self.logdet + logdet.sum(-1) # [B x *]
        
        return x
        
    def log_prob(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Compute the log probabilities at x.

        Parameters
        ----------
        x : Tensor [B x * x D]
            Point at which pdf is to be evaluated.
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.

        Returns
        -------
        Tensor [B x *]
            Log probabilities at x.
        '''
        z = self.forward(x, cond_var)
        
        log_prob_z = -0.5 * math.log(2 * math.pi) -0.5 * z**2
        log_prob_x = log_prob_z.sum(-1) + self.log_det()

        return log_prob_x
    
    def log_det(self) -> Tensor:
        '''
        Compute the log determinant of the Jacobian.
        '''
        return self.logdet
    
    def sample(self, size: list=[1]) -> Tensor:
        '''
        Sample from the learned distribution.
        '''
        raise NotImplementedError

from util import Init_random_seed as init_random_seed







                

        

                

        


