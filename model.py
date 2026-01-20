import torch
import torch.nn as nn
import pdb
from layers import *
from util import *

##CLIF
class CLIFModel(nn.Module):
    def __init__(self,configs):
        super(CLIFModel,self).__init__()
        self.rand_seed=configs['seed']

        self.label_emb=nn.Parameter(torch.eye(configs['num_classes']),requires_grad=False)
        self.label_edge=nn.Parameter(torch.eye(configs['num_classes']),requires_grad=False)

        self.hidden=768
        self.creation1=nn.MultiLabelSoftMarginLoss(reduction='none')
        self.creation2=LinkPredictionLoss_cosine()

        self.GIN_encoder=GIN(configs['num_layers'],input_size=configs['num_classes'],out_size=configs['class_emb_size'],
                                hidden_list=configs['hidden_list'])

        self.FD_model=FDModel(input_x_size=configs['input_x_size'],input_y_size=configs['class_emb_size'],hidden_size=self.hidden,out_size=self.hidden,
                                in_layers1=configs['in_layers'],out_layers=1,nonlinearity='leaky_relu',drop_ratio=0.2,negative_slope=0.1,batchNorm=False)
                                
        self.cls_conv=nn.Conv1d(configs['num_classes'],configs['num_classes'],self.hidden,groups=configs['num_classes'])
        self.reset_parameters()
        
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_emb)
        self.GIN_encoder.reset_parameters()
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()
        
    def get_config_optim(self):
        return [{'params': self.GIN_encoder.parameters()},
                {'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]
    def forward(self,x):
        label_emb=self.GIN_encoder(self.label_emb,self.label_edge)

        x=self.FD_model(x,label_emb)
        # print("cls_conv(x).shape:", self.cls_conv(x).shape)
        out=self.cls_conv(x).squeeze(dim=2)
        return out,label_emb

    def loss_function_train(self,outputs,label):
        loss,loss_matrix=self._compute_loss(*outputs,label)
        # print(loss_matrix.shape)
        return {"Loss": loss, "loss_matrix": loss_matrix.detach().cpu()}

    def _compute_loss(self,output,emb,label):
        adj=self.label_edge.data + torch.eye(self.label_edge.data.size(0),
                                                        dtype=self.label_edge.data.dtype,
                                                        device=self.label_edge.data.device)
       

        loss_per_sample_label = F.binary_cross_entropy_with_logits(output, label, reduction='none')
        # print(loss_per_sample_label.shape)
        reg_loss = 1e-3*self.creation2(emb,adj)
        loss = loss_per_sample_label.mean() + reg_loss
        return loss,loss_per_sample_label
    def predict(self,x,print_predictions=False, sample_ids=None, label_names=None):
        self.eval()
        with torch.no_grad():
            label_emb=self.GIN_encoder(self.label_emb,self.label_edge)

            x=self.FD_model(x,label_emb)
            out=self.cls_conv(x).squeeze(dim=2).sigmoid_()

            # if print_predictions and sample_ids is not None and label_names is not None:
            #     self.print_sample_predictions(out, sample_ids, label_names)
        return label_emb,out
    def custom_multilabel_soft_margin_loss(self,preds, targets, weights):
        individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
        weights_tensor = torch.tensor(weights, dtype=individual_losses.dtype, device=individual_losses.device)
        weighted_losses = individual_losses * weights_tensor
        return weighted_losses

    



##DELA
class DELA(nn.Module):
    def __init__(self, configs):
        super(DELA, self).__init__()
        self.configs = configs
        init_random_seed(self.configs['seed'])
        
        # Embedding function
        self.encoder = MLP(configs['in_features'], 256, [256, 512], False,
                           configs['drop_ratio'], "relu")
        self.fc_mu = nn.Linear(256, configs['latent_dim'])
        
        # Standard deviation function to parametrize the noise distribution 
        # (share the first three layers with the embedding function)
        self.fc_logvar = nn.Linear(256, configs['latent_dim'])
        
        # Function to parametrize the binary Concrete gates
        self.logit = nn.Parameter(torch.randn(configs['num_classes'], configs['latent_dim']))
        self.scale_layer = nn.Linear(configs['latent_dim'], configs['latent_dim'])
        
        # Classifiers
        self.decoder = MLP(configs['in_features']+configs['latent_dim'], 512,
                           [256], False, nonlinearity="relu")
        self.classifier = nn.Conv1d(configs['num_classes'], configs['num_classes'], 512,
                                    groups=configs['num_classes'])
        
        # Move model to the right device for consistent initialization
        self.to(configs['device'])
        self.training=True
        
        self.reset_parameters()

        
    def reset_parameters(self):
        init_random_seed(self.configs['seed'])
        self.encoder.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()
        self.decoder.reset_parameters()
        self.classifier.reset_parameters()
        self.logit.data.uniform_(-10, 10)
        self.scale_layer.reset_parameters()
        nn.init.constant_(self.scale_layer.bias, 2.0)
    
    def get_config_optim(self):
        return [{'params': self.encoder.parameters()},
                {'params': self.fc_mu.parameters()},
                {'params': self.fc_logvar.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.logit, 'lr': self.configs['lr_ratio']*self.configs['lr']},
                {'params': self.scale_layer.parameters(), 'lr': self.configs['lr_ratio']*self.configs['lr']}]
    
    def forward(self, input: Tensor) -> Tuple[Tensor, ...]:
        # Obtain latent representation of data and standard deviation of the noise distribution [B x D]
        z, n_logvar = self._encode(input)#[B,D]
        
        # Sample the indicator vector of non-informative features for each class label from binary Concrete gates
        if self.training:
            logit = self.scale_layer(self.logit) # [Q x D]
            samples = gumbel_sigmoid(logit, tau=self.configs['tau'], gumbel_noise=True, hard=True) # [Q x D]
             # For numerical stability when calculating the KL-divergence and smoother decision boundary
            #samples = samples.clamp(min=self.configs['off_noise']).detach() + samples - samples.detach()
        else:
            samples = None
            
        # Perturb latent representation
        z_k = self._add_noise(z, n_logvar, samples) # [B x Q x D]
        
        # Classification
        preds = self._decode(z_k, input) # [B x Q]
        
        return z, n_logvar, samples, preds
    
    
    def loss_function_train(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, Kl_loss, Cls_loss,loss_per_entry = self._compute_loss(*preds, targets) 
        
        return {'Loss': Loss,
                'Kl_loss': Kl_loss,
                'Cls_loss': Cls_loss,
                "loss_matrix":loss_per_entry}
    
    def loss_function_eval(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, _, Cls_loss = self._compute_loss(*preds, targets)
        
        return {'Loss': Loss.detach().item(),
                'Cls_loss': Cls_loss.detach().item()}
    
    def predict(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad():
            # Obtain latent representation of data [B x D]
            x_mu, _ = self._encode(input)
            z_x = self._add_noise(x_mu, None, None) # [B x Q x D]
            
            # Classification
            pred_probs = self._decode(z_x, input).sigmoid_() # [B x Q]
            pred_labels = (pred_probs > 0.5).type_as(pred_probs) # [B x Q]
            
        return pred_labels, pred_probs
        
    def _encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar

    def _add_noise(self, z: Tensor, n_logvar: Tensor, samples: Tensor=None):
        if samples is not None:
            std = torch.exp(0.5 * n_logvar) # sigma = exp(0.5 * log(sigma^2))
            eps = torch.randn_like(std)
            z_k = z.unsqueeze(1) + samples.unsqueeze(0) * std.unsqueeze(1) * eps.unsqueeze(1) # [B x Q x D]
        else:
            z_k = z.unsqueeze(1).expand(-1, self.configs['num_classes'], -1) # [B x Q x D]
            
        return z_k
    
    def _decode(self, z: Tensor, input: Tensor) -> Tensor:
        # Original feature is incorporated for more stable training. Similar technique has been used in Conditional VAE and MPVAE
        z = self.decoder(torch.cat([input.unsqueeze(1).expand(-1, self.configs['num_classes'], -1), z],
                                   dim=2)) # [B x Q x D]
        preds = self.classifier(z).squeeze(2) # [B x Q]
        
        return preds
    
    def _compute_loss(self, z: Tensor, n_logvar: Tensor, samples: Tensor,
                      preds: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        # Classification loss
        Cls_loss = F.multilabel_soft_margin_loss(preds, targets) * targets.size(1)
        # print(Cls_loss)
        loss_per_entry = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')  # shape [N, C]
        # print(loss_per_entry)
        if samples is not None:
            Kl_loss = self._KL(z, n_logvar, samples)
            Loss = Cls_loss + self.configs['beta'] * Kl_loss

        else:
            Kl_loss = None
            Loss = Cls_loss
            
        return Loss, Kl_loss, Cls_loss,loss_per_entry
    
    def _loss_per_label(self, preds, targets):
        return F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    def _loss_per_label2(self, z: Tensor, n_logvar: Tensor, samples: Tensor,
                      preds: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:

        Cls_loss = torch.mean(F.multilabel_soft_margin_loss(preds, targets, reduction='none'), dim=1)

        if samples is not None:
            Kl_loss = self._KL1(z, n_logvar, samples)
            Loss = Cls_loss + self.configs['beta'] * Kl_loss

        else:
            Kl_loss = None
            Loss = Cls_loss
        return Loss

    def custom_multilabel_soft_margin_loss(self,preds, targets, weights):
#         individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
#         return individual_losses
        sigmoid_preds = torch.sigmoid(preds)
        weights_tensor = torch.tensor(weights, dtype=preds.dtype, device=preds.device)
        losses = - (targets * torch.log(sigmoid_preds + 1e-6) + (1 - targets) * torch.log(1 - sigmoid_preds + 1e-6))
        loss_per_sample =weights_tensor * losses.sum(dim=1)
        return loss_per_sample
         
    def _KL(self, z: Tensor, n_logvar: Tensor, samples: Tensor):
        z = z.unsqueeze(1)
        n_logvar = n_logvar.unsqueeze(1)
        samples = samples.unsqueeze(0)
        KL_mat = -n_logvar - 2*torch.log(samples+1e-6) - 1 + torch.exp(n_logvar)*samples**2 + z**2 # [B x Q x D]
        return torch.mean(0.5*torch.sum(KL_mat, dim=2))

    def _KL1(self, z: Tensor, n_logvar: Tensor, samples: Tensor):
        z = z.unsqueeze(1)
        n_logvar = n_logvar.unsqueeze(1)
        samples = samples.unsqueeze(0)
        KL_mat = -n_logvar - 2*torch.log(samples+1e-6) - 1 + torch.exp(n_logvar)*samples**2 + z**2 # [B x Q x D]
#         return torch.mean(0.5*torch.sum(KL_mat, dim=2))
        KL_mean_per_sample = torch.mean(KL_sum, dim=1) 
        return KL_mean_per_sample
    
    

class PACA(nn.Module):
    def __init__(self, configs):
        super(PACA, self).__init__()
        self.configs = configs
        init_random_seed(self.configs['rand_seed'])
        
        # Probabilistic autoencoder for features
        self.encoder = MLP(configs['in_features'], 256, [256, 512], False,
                           configs['drop_ratio'], "relu")
        self.fc_mu = nn.Linear(256, configs['latent_dim'])
        self.fc_logvar = nn.Linear(256, configs['latent_dim'])
        self.decoder = MLP(configs['latent_dim'], configs['in_features'],
                           [256, 512, 256], False, nonlinearity="relu",
                           with_output_nonlineartity=False)
        
        # Probabilistic autoencoder for labels
        self.label_encoder = MLP(configs['num_classes'], 256, [512], False, 
                                 configs['drop_ratio'], "relu")
        self.label_fc_mu = nn.Linear(256, configs['latent_dim'])
        self.label_fc_logvar = nn.Linear(256, configs['latent_dim'])
        self.label_decoder = MLP(configs['latent_dim'], 512, [256], False,
                                 nonlinearity="relu")
        self.label_classifier = nn.Linear(512, configs['num_classes'])
        
        # Probabilistic prototypes via normalizing flows
        self.label_encodings = nn.Parameter(torch.eye(configs['num_classes']).unsqueeze(0),
                                            requires_grad=False)
        base_dist = torch.distributions.normal.Normal(torch.zeros(configs['latent_dim']).to(configs['device']),
                                                      torch.ones(configs['latent_dim']).to(configs['device']))
        self.pos_prototypes = self._create_normalizing_flows(base_dist)
        self.neg_prototypes = self._create_normalizing_flows(base_dist)

        # Instance-conditional mapping
        self.ins_map = MLP(configs['in_features']+configs['latent_dim'], configs['latent_dim'],
                           [256, 256], False, nonlinearity="relu",
                           with_output_nonlineartity=False)
        
        # Move model to the right device for consistent initialization
        self.to(configs['device'])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init_random_seed(self.configs['rand_seed'])
        self.encoder.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()
        self.label_encoder.reset_parameters()
        self.label_fc_mu.reset_parameters()
        self.label_fc_logvar.reset_parameters()
        self.decoder.reset_parameters()
        self.label_decoder.reset_parameters()
        self.label_classifier.reset_parameters()
        self.ins_map.reset_parameters()
        self.pos_prototypes.reset_parameters()
        self.neg_prototypes.reset_parameters()
    
    def get_config_optim(self):
        return [{'params': self.encoder.parameters()},
                {'params': self.fc_mu.parameters()},
                {'params': self.fc_logvar.parameters()},
                {'params': self.label_encoder.parameters()},
                {'params': self.label_fc_mu.parameters()},
                {'params': self.label_fc_logvar.parameters()},
                {'params': self.pos_prototypes.parameters()},
                {'params': self.neg_prototypes.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.label_decoder.parameters()},
                {'params': self.label_classifier.parameters()},
                {'params': self.ins_map.parameters()}]
    
    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor, ...]:
        # Probabilistic representation of instance and label vector [B x D]
        x_mu, x_logvar = self._encode(input)
        y_mu, y_logvar = self._label_encode(target)
        if self.training:
            z_x = self._reparameterize(x_mu, x_logvar)
            z_y = self._reparameterize(y_mu, y_logvar)
        else:
            z_x = x_mu
            z_y = y_mu
        
        # Latent space regularization
        # KL[q(z|x)||q(z|y)]
        kl_div = torch.mean(0.5*torch.sum(y_logvar-x_logvar-1+torch.exp(x_logvar-y_logvar)
                                          +(y_mu-x_mu)**2/(torch.exp(y_logvar)
                                          +self.configs['eps']), dim=1))
        preds_y = self._label_decode(z_y)
        
        # Instance-conditional mapping for more stable training. Similar technique has been used in Conditional VAE and MPVAE
        z_x = self.ins_map(torch.cat([input, z_x], dim=1))
        
        # Reconstuction
        recons = self._decode(z_x)
        
        # Instance's log probs on positive/negative prototypes of each class label [B x 2 x Q]
        log_ins_class_probs = self._log_density_proto(z_x)
        # Distances between instance and prototypes, i.e. label-specific features [B x 2 x Q]
        # [-KL[q(z|x)||p(z|N^j)], -KL[q(z|x)||p(z|P^j)]] is equivalent to [E_z[p(z|N^j)], E_z[p(z|P^j)]] in implementation
        dists_x = log_ins_class_probs
        
        return input, kl_div, recons, preds_y, dists_x
    
    def training_start(self, train_dataloader):
        '''
        Prepare for training.
        '''
        self.iters_per_epoch = len(train_dataloader)
    
    def loss_function_train(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, Recons_loss, Reg_loss, Cls_loss,per_loss = self._compute_loss(*preds, targets) 

        return {'Loss': Loss,
                'Recons_loss': Recons_loss,
                'Reg_loss': Reg_loss,
                'Cls_loss': Cls_loss,
                'loss_matrix':per_loss}
    
    def predict(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad():
            # Probabilistic representation of instance [B x D]
            x_mu, _ = self._encode(input)
            
            # Instance-conditional mapping for more stable training. Similar technique has been used in Conditional VAE and MPVAE
            z_x = self.ins_map(torch.cat([input, x_mu], dim=1))
            
            # Instance's log probs on positive/negative prototypes of each class label [B x 2 x Q]
            log_class_probs = self._log_density_proto(z_x)
            # Distances between instance and prototypes, i.e. label-specific features [B x 2 x Q]
            # [-KL[q(z|x)||p(z|N^j)], -KL[q(z|x)||p(z|P^j)]] is equivalent to [E_z[p(z|N^j)], E_z[p(z|P^j)]] in implementation
            dists = log_class_probs
            
            # Classification with parameter-free classifiers
            pred_probs = torch.softmax(dists, dim=1)[:, 1, :] # prob for label occurrence
            pred_labels = (pred_probs > 0.5).type_as(pred_probs)
            # print(pred_probs)
            # print(pred_labels)
        return pred_labels, pred_probs
    
    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = torch.optim.Adam(self.get_config_optim(), lr=self.configs['lr'],
                                     weight_decay=self.configs['weight_decay'])
        if self.configs['lr_scheduler'] == 'step_epoch':
            scheduler = StepLRScheduler(optimizer,
                                        decay_t=self.configs['scheduler_decay_epoch'],
                                        decay_rate=self.configs['scheduler_decay_rate'],
                                        t_in_epoch=True,
                                        iters_per_epoch=self.iters_per_epoch,
                                        warmup_t=self.configs['scheduler_warmup_epoch'])
        else:
            scheduler = None
            
        return optimizer, scheduler
    
    def _create_normalizing_flows(self, base_dist):
        # Create diffeomorphisms
        flow_trans = []
        flow_trans.append(CondNAF(self.configs['latent_dim'], self.configs['num_classes'], [256]))
        flow_trans.append(CondAF(self.configs['latent_dim'], self.configs['num_classes'],
                                 identity_init=True))
        
        return CondTDist(base_dist, flow_trans)
        
    def _encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Encode the input by passing through the encoder network and return the
        latent codes.

        Parameters
        ----------
        input : Tensor
            Input tensor to encode.

        Returns
        -------
        Tuple(Tensor, Tensor)
            Mean and log variance parameters of the latent Gaussian distribution.
        '''
        result = self.encoder(input)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar
    
    def _label_encode(self, target: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Encode the input by passing through the encoder network and return the
        latent codes.

        Parameters
        ----------
        target : Tensor
            Input tensor to encode.

        Returns
        -------
        Tuple(Tensor, Tensor)
            Mean and log variance parameters of the latent Gaussian distribution.
        '''
        result = self.label_encoder(target)
        mu = self.label_fc_mu(result)
        logvar = self.label_fc_logvar(result)
        
        return mu, logvar
    
    def _decode(self, z: Tensor) -> Tensor:
        '''
        Decode the latent codes by passing through the decoder network.

        Parameters
        ----------
        z : Tensor [B x D]
            Latent codes to decode.

        Returns
        -------
        Tensor
            Reconstruction.
        '''
        return self.decoder(z)
    
    def _label_decode(self, z: Tensor) -> Tensor:
        '''
        Decode the latent codes by passing through the decoder network.

        Parameters
        ----------
        z : Tensor [B x D]
            Latent codes to decode.

        Returns
        -------
        Tensor
            Reconstruction.
        '''
        return self.label_classifier(self.label_decoder(z))
    
    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        '''
        Reparameterize trick to sample from N(mu, var).

        Parameters
        ----------
        mu : Tensor [B x D]
            Mean of the latent Gaussian.
        logvar : Tensor [B x D]
            Log variance of the latent Gaussian.

        Returns
        -------
        Tensor [B x D]
            Sampled latent codes.
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps * std + mu
    
    def _log_density_proto(self, x: Tensor) -> Tensor:
        '''
        Compute instance's log probability on positive/negative prototypes of each class label.

        Parameters
        ----------
        x : Tensor [B x D]
            Point at which density is to be evaluated.

        Returns
        -------
        Tensor [B x 2 x Q]
            log density at x.
        '''
        x_temp = x.unsqueeze(1)
        pos_log_density = self.pos_prototypes.log_prob(x_temp, self.label_encodings) # [B x Q]
        neg_log_density = self.neg_prototypes.log_prob(x_temp, self.label_encodings) # [B x Q]
        
        return torch.stack([neg_log_density, pos_log_density], dim=1)
    
    def _compute_loss(self, input: Tensor, kl_div: Tensor, recons: Tensor,
                      preds_y: Tensor, dists_x: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        batch_size = input.size(0)
        
        loss_per_sample_label = F.binary_cross_entropy_with_logits(preds_y,targets,reduction='none')
        # print(loss_per_sample_label.shape)
        # Reconstruction loss
        if self.configs['binary_data']:
            Recons_loss = F.binary_cross_entropy_with_logits(recons, input, reduction='sum') / batch_size
        else:
            Recons_loss = F.mse_loss(recons.sigmoid(), input, reduction='sum') / batch_size
        
        # Latent space regularization loss
        Reg_loss = kl_div + F.multilabel_soft_margin_loss(preds_y, targets) * targets.size(1)
        
        # Classification loss
        Cls_loss = F.cross_entropy(dists_x, targets.long()) * targets.size(1)
        
        # Overall loss
        Loss = Recons_loss + self.configs['gamma'] * Reg_loss + self.configs['alpha'] * Cls_loss
        
        return Loss, Recons_loss, Reg_loss, Cls_loss,loss_per_sample_label
    def _loss_per_label(self, preds, targets):
        return F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    def custom_multilabel_soft_margin_loss(self,preds, targets, weights):
#         individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
#         return individual_losses
        sigmoid_preds = torch.sigmoid(preds)
        weights_tensor = torch.tensor(weights, dtype=preds.dtype, device=preds.device)
        losses = -weights_tensor * (targets * torch.log(sigmoid_preds + 1e-6) + (1 - targets) * torch.log(1 - sigmoid_preds + 1e-6))
        loss_per_sample = losses.sum(dim=1)
        return loss_per_sample


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def normal_cdf(x):
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# ------------------------
# MPVAE PyTorch (CLIF-friendly interface)
# ------------------------
class MPVAE(nn.Module):
    def __init__(self, configs):
        super(MPVAE, self).__init__()
        # configs (with defaults)
        self.device = configs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.label_dim = configs['num_classes']
        self.feat_dim = configs.get('input_x_size', configs.get('in_features', 128))
        self.latent_dim = configs.get('latent_dim', max(1, self.feat_dim // 2))   # z in the VAE
        self.z_dim = configs.get('z_dim', max(16, self.latent_dim))               # residual noise dim (rank)
        self.n_train_sample = configs.get('n_train_sample', 100)
        self.n_test_sample = configs.get('n_test_sample', 1000)
        # loss coeffs
        self.nll_coeff = configs.get('nll_coeff', 0.1)
        self.c_coeff = configs.get('c_coeff', 200.0)
        self.l2_coeff = configs.get('l2_coeff', 1.0)
        self.scale_coeff = configs.get('scale_coeff', 1.0)
        self.eps = configs.get('eps', 1e-6)

        # ---------- Encoders / Decoders ----------
        # Label encoder: input = concat(x, y) -> fe_mu, fe_logvar
        hid = max(128, self.feat_dim + self.label_dim)
        self.label_encoder_fc1 = nn.Linear(self.feat_dim + self.label_dim, 512)
        self.label_encoder_fc2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, self.latent_dim)
        self.fe_logvar = nn.Linear(256, self.latent_dim)

        # Feature encoder: input = x -> fx_mu, fx_logvar
        self.feat_encoder_fc1 = nn.Linear(self.feat_dim, 256)
        self.feat_encoder_fc2 = nn.Linear(256, 256)
        self.fx_mu = nn.Linear(256, self.latent_dim)
        self.fx_logvar = nn.Linear(256, self.latent_dim)

        # Label decoder (from c_fe_sample)
        self.fd_1 = nn.Linear(self.feat_dim + self.latent_dim, 256)
        self.fd_2 = nn.Linear(256, 512)
        self.label_mp_mu_layer = nn.Linear(512, self.label_dim)

        # Feature decoder (from c_fx_sample)
        self.fd_x_1 = nn.Linear(self.feat_dim + self.latent_dim, 256)
        self.fd_x_2 = nn.Linear(256, 512)
        self.feat_mp_mu_layer = nn.Linear(512, self.label_dim)

        # # Low-rank residual covariance parameter: r_sqrt_sigma shape (label_dim, z_dim)
        # r_init = np.random.uniform(-np.sqrt(6.0/(self.label_dim + self.z_dim)),
        #                            np.sqrt(6.0/(self.label_dim + self.z_dim)),
        #                            (self.label_dim, self.z_dim)).astype(np.float32)
        r_init = np.random.uniform(-0.01, 0.01, (self.label_dim, self.z_dim)).astype(np.float32)

        self.r_sqrt_sigma = nn.Parameter(torch.tensor(r_init, device=self.device), requires_grad=True)

        # Optional small L2 regularization placeholder (we will compute param norms in loss)
        # Move to device
        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        # simple initializer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def get_config_optim(self):
        return [{'params': self.parameters()}]

    # ------------------------
    # Forward: compute prior (feature encoder outputs).
    # We keep forward(x) minimal so training loop calling net(x) remains unchanged.
    # Return a tuple that loss_function_train will use together with labels.
    # ------------------------
    def forward(self, x):
        """
        x: [batch, feat_dim]
        returns tuple (x, fx_mu, fx_logvar)
        """
        x = x.to(self.device)
        # feature encoder
        h1 = F.relu(self.feat_encoder_fc1(x))
        h2 = F.relu(self.feat_encoder_fc2(h1))
        fx_mu = self.fx_mu(h2) * self.scale_coeff
        fx_logvar = self.fx_logvar(h2) * self.scale_coeff

        # we return x for later concatenation with label-encoder sampling
        return (x, fx_mu, fx_logvar)

    # ------------------------
    # Loss function expected by training pipeline
    # inputs:
    #   outputs = net(x)  (the tuple returned above)
    #   labels: tensor [batch, label_dim] (float 0/1)
    # returns dictionary with 'Loss' and 'loss_matrix' (per-sample-per-label)
    # ------------------------
    def loss_function_train(self, outputs, labels):
        """
        outputs: (x, fx_mu, fx_logvar)
        labels: [batch, label_dim] float 0/1
        """
        x, fx_mu, fx_logvar = outputs
        x = x.to(self.device)
        labels = labels.to(self.device).float()
        batch = x.size(0)

        # ---------- label encoder (q(z|x,y)) ----------
        xy = torch.cat([x, labels], dim=1)
        h1 = F.relu(self.label_encoder_fc1(xy))
        h2 = F.relu(self.label_encoder_fc2(h1))
        fe_mu = self.fe_mu(h2) * self.scale_coeff
        fe_logvar = self.fe_logvar(h2) * self.scale_coeff

        # sample from q(z|x,y) and p(z|x)
        fe_sample = reparameterize(fe_mu, fe_logvar)   # [batch, latent_dim]
        fx_sample = reparameterize(fx_mu, fx_logvar)   # [batch, latent_dim]

        # label decoder branch
        c_fe_sample = torch.cat([x, fe_sample], dim=1)    # [batch, feat+latent]
        d1 = F.relu(self.fd_1(c_fe_sample))
        d2 = F.relu(self.fd_2(d1))
        label_mp_mu = self.label_mp_mu_layer(d2)          # [batch, label_dim]

        # feature decoder branch (for test-time prediction)
        c_fx_sample = torch.cat([x, fx_sample], dim=1)
        dx1 = F.relu(self.fd_x_1(c_fx_sample))
        dx2 = F.relu(self.fd_x_2(dx1))
        feat_mp_mu = self.feat_mp_mu_layer(dx2)          # [batch, label_dim]

        # ---------- low-rank residual covariance ----------
        # sigma = r_sqrt_sigma @ r_sqrt_sigma.T ; covariance = sigma + I
        B = self.r_sqrt_sigma.t()   # [z_dim, label_dim]

        # ---------- sample noise for Probit ----------
        n_sample = self.n_train_sample
        # noise shape: [n_sample, batch, z_dim]
        noise = torch.randn((n_sample, batch, self.z_dim), device=self.device)

        # sample_r (label branch) shape: [n_sample, batch, label_dim]
        sample_r = torch.matmul(noise, B) + label_mp_mu.unsqueeze(0)  # broadcast add
        sample_r_x = torch.matmul(noise, B) + feat_mp_mu.unsqueeze(0)

        # Gaussian CDF -> probabilities (clipped)
        E = normal_cdf(sample_r).clamp(min=self.eps*0.5, max=1.0-self.eps*0.5)           # [n_sample, batch, label_dim]
        E_x = normal_cdf(sample_r_x).clamp(min=self.eps*0.5, max=1.0-self.eps*0.5)

        # ---------- compute NLL (BCE-like) using importance sampling (log-sum-exp trick) ----------
        # sample_nll shape [n_sample, batch, label_dim]
        sample_nll = - (torch.log(E) * labels.unsqueeze(0) + torch.log(1.0 - E) * (1.0 - labels.unsqueeze(0)))
        logprob = - torch.sum(sample_nll, dim=2)   # [n_sample, batch]

        maxlog, _ = torch.max(logprob, dim=0)   # [batch]
        Eprob = torch.mean(torch.exp(logprob - maxlog.unsqueeze(0)), dim=0)  # [batch]
        nll_loss = torch.mean(- torch.log(Eprob + 1e-12) - maxlog)  # scalar

        # feature branch NLL
        sample_nll_x = - (torch.log(E_x) * labels.unsqueeze(0) + torch.log(1.0 - E_x) * (1.0 - labels.unsqueeze(0)))
        logprob_x = - torch.sum(sample_nll_x, dim=2)
        maxlog_x, _ = torch.max(logprob_x, dim=0)
        Eprob_x = torch.mean(torch.exp(logprob_x - maxlog_x.unsqueeze(0)), dim=0)
        nll_loss_x = torch.mean(- torch.log(Eprob_x + 1e-12) - maxlog_x)

        # ---------- Ranking loss (pairwise) ----------
        c_loss = self._build_multi_classify_loss(E, labels)
        c_loss_x = self._build_multi_classify_loss(E_x, labels)

        # ---------- KL between q(z|x,y) and p(z|x) ----------
        # using formula from TF code: mean 0.5 * sum((fx_logvar-fe_logvar)-1+exp(fe_logvar-fx_logvar) + (fx_mu-fe_mu)^2 / (exp(fx_logvar)+eps))
        term1 = (fx_logvar - fe_logvar)
        term2 = torch.exp(fe_logvar - fx_logvar)
        denom = torch.exp(fx_logvar) + self.eps
        term3 = (fx_mu - fe_mu) ** 2 / denom
        kl_per_sample = 0.5 * torch.sum(term1 - 1.0 + term2 + term3, dim=1)  # [batch]
        kl_loss = torch.mean(kl_per_sample)

        # ---------- per-entry loss matrix for tracking (same as CLIF style) ----------
        loss_per_entry = F.binary_cross_entropy_with_logits(label_mp_mu, labels, reduction='none')  # [batch, label_dim]
        # print(loss_per_entry.shape)
        # note: we use label_mp_mu logits here for per-entry losses (consistent with TF's sample nll aggregate)
        # Alternatively you may use preds from feature branch; CLIF uses model logits before sigmoid.

        # ---------- l2 regularization (optional) ----------
        l2_loss = 0.0
        # if self.l2_coeff > 0:
        #     l2_loss = 0.0
        #     for p in self.parameters():
        #         l2_loss = l2_loss + (p ** 2).sum()
        #     l2_loss = 0.5 * l2_loss  # scale (you can adjust)

        # ---------- total loss composition (mirror TF) ----------
        total_loss = self.l2_coeff * l2_loss \
                     + (nll_loss + nll_loss_x) * self.nll_coeff \
                     + (c_loss + c_loss_x) * self.c_coeff \
                     + kl_loss * 1.1

        return {'Loss': total_loss, 'loss_matrix': loss_per_entry.detach().cpu()}

    def _build_multi_classify_loss(self, E, labels):
        """
        E: [n_sample, batch, label_dim]  (probabilities phi(sample_r))
        labels: [batch, label_dim] (0/1)
        Implementation follows the TensorFlow logic (pairwise c_i - c_k, exp(-5 * diff), mask where y_i=1 & y_k=0)
        """
        n_sample, batch, Q = E.shape
        # boolean masks
        y_i = (labels == 1)            # [batch, Q]
        y_not_i = (labels == 0)        # [batch, Q]

        # pairwise truth_matrix: [batch, Q, Q], True where (label_i == 1 AND label_k == 0)
        truth = (y_i.unsqueeze(2) & y_not_i.unsqueeze(1)).float()  # [batch, Q, Q]

        # pairwise differences: build [n_sample, batch, Q, Q]
        # For E (n_sample,batch,Q):
        a = E.unsqueeze(3)   # [n_sample,batch,Q,1]
        b = E.unsqueeze(2)   # [n_sample,batch,1,Q]
        sub = a - b          # [n_sample,batch,Q,Q]

        exp_mat = torch.exp(-5.0 * sub)  # [n_sample,batch,Q,Q]

        # broadcast truth to [n_sample,batch,Q,Q]
        truth_b = truth.unsqueeze(0)   # [1,batch,Q,Q] -> broadcast

        sparse = exp_mat * truth_b    # zero-out pairs not needed
        # sum over pairs (i,k)
        sums = torch.sum(sparse, dim=(2, 3))   # [n_sample, batch]

        # normalizers: per-sample y_i_sizes * y_i_bar_sizes  -> [batch]
        y_i_sizes = torch.sum(y_i.float(), dim=1)       # [batch]
        y_i_bar_sizes = torch.sum(y_not_i.float(), dim=1)  # [batch]
        normalizers = y_i_sizes * y_i_bar_sizes + 1e-12

        # loss shape [n_sample, batch], divide element-wise (broadcast normalizers to [1,batch])
        loss = sums / (5.0 * normalizers.unsqueeze(0))
        # sanitize
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        loss = loss.mean(dim=0)   # mean over n_sample -> [batch]
        loss = loss.mean()        # scalar
        return loss

    # ------------------------
    # Predict: use feature branch prior p(z|x) and average across many samples
    # returns (None, pred_probs) to mimic CLIF predict signature
    # ------------------------
    def predict(self, x):
        """
        x: [batch, feat_dim]
        returns: (None, pred_probs) where pred_probs [batch, label_dim]
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            # encode
            h1 = F.relu(self.feat_encoder_fc1(x))
            h2 = F.relu(self.feat_encoder_fc2(h1))
            fx_mu = self.fx_mu(h2) * self.scale_coeff
            fx_logvar = self.fx_logvar(h2) * self.scale_coeff

            batch = x.size(0)
            B = self.r_sqrt_sigma.t()  # [z_dim, label_dim]
            n_sample = self.n_test_sample
            noise = torch.randn((n_sample, batch, self.z_dim), device=self.device)
            # sample fx_sample many times and create feat_mp_mu for each sample
            # but simpler: sample fx_sample once per sample index and reuse (consistent with TF approach)
            fx_sample = reparameterize(fx_mu, fx_logvar)   # [batch, latent_dim]
            c_fx_sample = torch.cat([x, fx_sample], dim=1)
            dx1 = F.relu(self.fd_x_1(c_fx_sample))
            dx2 = F.relu(self.fd_x_2(dx1))
            feat_mp_mu = self.feat_mp_mu_layer(dx2)        # [batch, label_dim]

            sample_r_x = torch.matmul(noise, B) + feat_mp_mu.unsqueeze(0)   # [n_sample,batch,label_dim]
            E_x = normal_cdf(sample_r_x).clamp(min=self.eps*0.5, max=1.0-self.eps*0.5)
            # average across samples
            indiv_prob = torch.mean(E_x, dim=0)   # [batch, label_dim]

        return None, indiv_prob.cpu()
