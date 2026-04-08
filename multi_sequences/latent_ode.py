###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import sklearn as sk
import numpy as np
#import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

import rnn_utils as utils
from rnn_utils import get_device
from encoder_decoder import *
#from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
from run_dnerf_helpers import LatentNetwork, get_embedder
#from lib.base_models import VAE_Baseline
import random
import torch.nn.functional as F

class LatentODE(nn.Module):
    def __init__(self, args, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
        z0_prior, device, obsrv_std = None, 
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr = False,
        num_frames=80,
        latent_embedder_out_dim=512,
        latent_embedder = None,
        embed_angle=None,
        embed_vel = None,
        angle_linear_layer=None,
        decoder_pose = None,
        decoder_vel = None,
        z0_encoder_type = "odernn",
        linear = None,
        encoder_z0_vel = None,
        concat_net = None):

        super(LatentODE, self).__init__()
        self.latent_embedder_out_dim = latent_embedder_out_dim

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        # self.diffeq_solver_1 = diffeq_solver[1]
        self.device = device
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc
        self.num_frames = num_frames
        self.z0_prior = z0_prior
        self.latent_dim = latent_dim
        # print("Num Frames: ", num_frames)
        self.latent_net = LatentNetwork(input_size=1, 
                                        latent_size=latent_embedder_out_dim)
        self.static_latent_net = LatentNetwork(input_size=1, 
                                        latent_size=latent_embedder_out_dim)
        self.embed_angle = embed_angle
        self.embed_vel = embed_vel
        self.angle_linear_layer = angle_linear_layer
        self.decoder_pose = decoder_pose
        self.decoder_vel = decoder_vel
        self.z0_encoder_type = z0_encoder_type
        self.linear = linear
        self.encoder_z0_vel = encoder_z0_vel
        self.concat_net = concat_net
        self.static_background = args.static_background

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, latent_truth_time_steps, vel=None,
        latent_time_steps_to_pred=None, angle=None, mask = None, n_traj_samples = 1, run_backwards = True, mode = None,
        warmup=False):
        
        latent_time_input = latent_truth_time_steps
        if len(latent_time_input.shape) == 0:
            latent_time_input = torch.unsqueeze(latent_time_input, 0)
        static_latent = self.static_latent_net(torch.tensor([0])).float().squeeze()

        if warmup:
            # return latent_embeddings, None, None
            return static_latent, None, None

        latent_embeddings = self.latent_net(torch.tensor([0])).float().squeeze()    

        latent_embeddings = latent_embeddings.unsqueeze(0).unsqueeze(0)
        angle = self.embed_angle(angle).squeeze()
        angle = angle.unsqueeze(0).unsqueeze(0)
        
        vel = vel.unsqueeze(0).unsqueeze(0)
        
        if self.z0_encoder_type == 'mlp':
            h = angle
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                if i < 8:
                    h = F.relu(h)
                if i == 4:
                    h = torch.cat([angle, h], -1)
                    
            first_point_enc_aug = h




        elif self.z0_encoder_type == 'linear':
            first_point_enc_aug = self.encoder_z0(angle)
        
        first_point_enc_aug = torch.cat([first_point_enc_aug, vel, latent_embeddings], dim=-1)

        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        pred_x = torch.squeeze(self.decoder(sol_y))
        if self.static_background:
            pred_x = pred_x + static_latent

        pred_pose = torch.squeeze(self.decoder_pose(sol_y))

        pred_vel = torch.squeeze(self.decoder_vel(sol_y))[:-1]

        return pred_x, pred_pose, pred_vel

    def forward(self, batch_dict, warmup=False, freeze=False, div=False):

        
        angle = batch_dict["angle"].to(self.device).squeeze()
        angle = angle[-1]
        angle_to_pred = batch_dict["angle_to_pred"].to(self.device).squeeze()

        vel_org = angle_to_pred[0] - angle.squeeze()
        # vel = torch.dot(vel_org, torch.tensor([1.,0.,0.]))
        vel = self.embed_vel(vel_org)

        angle_to_pred1 = angle_to_pred[:-1]
        angle_to_pred2 = angle_to_pred[1:]

        vel_to_pred = angle_to_pred2 - angle_to_pred1
        vel_latent_to_pred = self.embed_vel(vel_to_pred)
        ## Just 2D pose
        # angle = angle[...,:-1]
        # angle_to_pred = angle_to_pred[...,:-1]

        angle_latent_to_pred = self.embed_angle(angle_to_pred)

        
        batch_dict["times"] = batch_dict["times"].to(self.device)
        batch_dict["times_to_pred"] = batch_dict["times_to_pred"].to(self.device)


        if warmup:
            seen = 0
            self.latent_net.requires_grad = False
            self.static_latent_net.requires_grad = True
            self.diffeq_solver.requires_grad = False
        else:

            seen = np.random.choice(torch.arange(0,batch_dict["times_to_pred"].shape[-1]).cpu().detach().numpy())

            self.latent_net.requires_grad = True
            self.static_latent_net.requires_grad = False
            self.diffeq_solver.requires_grad = True

        # latent_truth_time_steps = batch_dict["times"]
        latent_truth_time_steps = torch.tensor([[0,1]])
        truth_time_steps = torch.squeeze(latent_truth_time_steps)


        latent_time_steps_to_pred = batch_dict["times_to_pred"]
        
        time_steps_to_predict = torch.squeeze(latent_time_steps_to_pred)#[seen]

        time_steps_to_predict = time_steps_to_predict - time_steps_to_predict[0]

        if div:
            # time_steps_to_predict = time_steps_to_predict.flip(0)
            time_steps_to_predict.requires_grad = True ## ONLY FOR DIV (NOT USED)
            # time_steps_to_predict1 = time_steps_to_predict.flip(0)

        
        if len(time_steps_to_predict.shape) == 0:
            time_steps_to_predict = torch.unsqueeze(time_steps_to_predict, 0)
        # if len(time_steps_to_predict1.shape) == 0:
        #     time_steps_to_predict1 = torch.unsqueeze(time_steps_to_predict1, 0)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)


        latents, latents_pose, latents_vel = self.get_reconstruction(
            time_steps_to_predict=time_steps_to_predict,
            truth=None,
            truth_time_steps=truth_time_steps,
            vel=vel,
            latent_truth_time_steps=latent_truth_time_steps,
            angle = angle,
            mask=None,
            warmup = warmup)
        
        if warmup:
            #return latents[seen], seen, None, None  ## THIS
            return latents, seen, None, None, None, None, None
        divergence = None
        if div:
            latents_pose = latents_pose[:angle_latent_to_pred.squeeze().shape[0]]
            divergence = torch.autograd.grad(outputs=latents, inputs=time_steps_to_predict, grad_outputs=torch.ones_like(latents), create_graph=True, retain_graph=True)[0]

        return latents, seen, latents_pose, angle_latent_to_pred.squeeze(), divergence, latents_vel, vel_latent_to_pred.squeeze()

    
    def next_latent(self, latent, times_obs, times_to_pred, angle=None, loc=None, run_backwards=True, n_traj_samples=1, vel=0):

        if not angle:
            angle = loc
        truth_time_steps = torch.squeeze(times_obs)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        time_steps_to_predict = torch.squeeze(times_to_pred)

        static_latent = self.static_latent_net(torch.tensor([0])).float()

        latents = self.latent_net(torch.tensor([0])).float()
        latents = latents.unsqueeze(0).unsqueeze(0)

        angle = self.embed_angle(torch.tensor(angle))
        #angle = angle.unsqueeze(0)  ##  FOR MLP
        angle = angle.unsqueeze(0).unsqueeze(0)  ##FOR RNNODE
        #angle = torch.cat([angle, latents], dim=-1)  ## For angle+latent --> MLP

        vel = torch.tensor(vel)
        vel = self.embed_vel(vel)
        vel = vel.unsqueeze(0).unsqueeze(0)

        if self.z0_encoder_type == 'mlp':
            h = angle
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                if i < 8:
                    h = F.relu(h)
                if i == 4:
                    h = torch.cat([angle, h], -1)
                    
            first_point_enc_aug = h

        
        elif self.z0_encoder_type == 'linear':
            first_point_enc_aug = self.encoder_z0(angle)


        #first_point_enc_aug = first_point_enc_aug.unsqueeze(0) ## FOR MLP
        
        
        first_point_enc_aug = torch.cat([first_point_enc_aug, vel, latents], dim=-1)
        # first_point_enc_aug = torch.cat([first_point_enc_aug, vel_enc, latents], dim=-1)

        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        #sol_y = torch.cat([sol_y, first_point_enc_aug.squeeze().repeat(1,1, sol_y.shape[2],1)], dim=-1)

        pred_x = self.decoder(sol_y).squeeze()# + static_latent
        if self.static_background:
            pred_x = pred_x + static_latent

        #pred_x = pred_x + torch.squeeze(latents)

        return torch.squeeze(pred_x), None


    def next_latent_batch(self, latent, times_obs, times_to_pred, angle=None, loc=None, run_backwards=True, n_traj_samples=1, vel=[0.,0.]):

        if not angle:
            angle=loc
        #truth_time_steps = torch.squeeze(times_obs)
        truth_time_steps = torch.tensor([0.], requires_grad=True)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        time_steps_to_predict = torch.squeeze(times_to_pred)
        time_steps_to_predict.requires_grad = True
        time_steps_to_predict1 = time_steps_to_predict.flip(0)
        
        
        angle_input = torch.tensor(angle)
        angle_input.requires_grad = True
        if latent==None:
            static_latent = self.static_latent_net(torch.tensor([0])).float()
            latents = self.latent_net(torch.tensor([0])).float()

            
            angle = self.embed_angle(angle_input).squeeze()
            if len(angle.shape) == 1:  
                truth = angle.unsqueeze(0).unsqueeze(0)
            else:
                truth = angle.unsqueeze(1)

            truth_w_mask = truth
            print(truth_w_mask.shape)
            
            latents = latents.repeat(angle.shape[0], 1)
            if len(latents.shape) == 1: 
                latents = latents.unsqueeze(0).unsqueeze(0)
            else:
                latents = latents.unsqueeze(0)

        if self.z0_encoder_type == 'mlp':
            h = truth_w_mask
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                if i < 8:
                    h = F.relu(h)
                if i == 4:
                    h = torch.cat([truth_w_mask, h], -1)
                    
            first_point_enc_aug = h
            
            #print("First Apoint Enc Aug: ", first_point_enc_aug.shape)
            ### [40,1,63] --> [1,40,63]
            first_point_enc_aug = first_point_enc_aug.permute(1,0,2)


        #first_point_enc_aug = torch.cat([latents, first_point_enc_aug], dim=-1)
        vel = torch.tensor(vel)
        vel.requires_grad = True
        vel = self.embed_vel(vel)
        vel = vel.repeat(angle.shape[0], 1)
        if len(vel.shape) == 1: 
            vel = vel.unsqueeze(0).unsqueeze(0)
        else:
            vel = vel.unsqueeze(0)

        first_point_enc_aug = torch.cat([first_point_enc_aug, vel, latents], dim=-1)

        ## Concat Layer
        # for i, l in enumerate(self.concat_net):
        #     first_point_enc_aug = self.concat_net[i](first_point_enc_aug)
        #     if i < 2:
        #         first_point_enc_aug = F.relu(first_point_enc_aug)
        ####
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict1)


        # sol_y_x = sol_y[..., -512:]
        # sol_y_pose = sol_y[..., :8]
        
        pred_x = torch.squeeze(self.decoder(sol_y)) + static_latent
        print("Pred X: ", pred_x.shape)
        divergences = []
        for i in range(pred_x.shape[0]):
            
            div = torch.autograd.grad(outputs=pred_x[i], inputs=time_steps_to_predict, grad_outputs=torch.ones_like(pred_x[i]), create_graph=True, retain_graph=True)[0]
            print("Grad: ", div.shape)
            divergences.append(div)

        # sol_y = sol_y.squeeze()
        # for i in range(sol_y.shape[0]):
            
        #     div = torch.autograd.grad(outputs=sol_y[i], inputs=time_steps_to_predict, grad_outputs=torch.ones_like(sol_y[i]), create_graph=True, retain_graph=True)[0]
        #     print("Grad: ", div.shape)
        #     divergences.append(div)
        
        # pred_pose = torch.squeeze(self.decoder_pose(sol_y))
        return torch.squeeze(pred_x), torch.stack(divergences), None
        # return sol_y.squeeze(), torch.stack(divergences), pred_pose


    def next_latent_batch_vel(self, latent, times_obs, times_to_pred, angle=None, loc=None, run_backwards=True, n_traj_samples=1, vel=[0.,0.]):

        if not angle:
            angle=loc
        #truth_time_steps = torch.squeeze(times_obs)
        truth_time_steps = torch.tensor([0.], requires_grad=True)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        time_steps_to_predict = torch.squeeze(times_to_pred)
        time_steps_to_predict.requires_grad = True
        time_steps_to_predict1 = time_steps_to_predict.flip(0)
        
        vel = torch.tensor(vel)
        vel.requires_grad = True
        vel = self.embed_vel(vel)
        if len(vel.shape) == 1: 
            vel = vel.unsqueeze(0).unsqueeze(0)
        else:
            vel = vel.unsqueeze(0)
        
        angle_input = torch.tensor(angle)
        angle_input.requires_grad = True
        if latent==None:
            static_latent = self.static_latent_net(torch.tensor([0])).float()
            latents = self.latent_net(torch.tensor([0])).float()

            
            angle = self.embed_angle(angle_input).squeeze()
            angle = angle.repeat(vel.shape[1], 1)
            if len(angle.shape) == 1:  
                truth = angle.unsqueeze(0).unsqueeze(0)
            else:
                truth = angle.unsqueeze(1)

            truth_w_mask = truth
            print(truth_w_mask.shape)
            
            latents = latents.repeat(angle.shape[0], 1)
            if len(latents.shape) == 1: 
                latents = latents.unsqueeze(0).unsqueeze(0)
            else:
                latents = latents.unsqueeze(0)

        if self.z0_encoder_type == 'mlp':
            h = truth_w_mask
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                if i < 8:
                    h = F.relu(h)
                if i == 4:
                    h = torch.cat([truth_w_mask, h], -1)
                    
            first_point_enc_aug = h
            
            #print("First Apoint Enc Aug: ", first_point_enc_aug.shape)
            ### [40,1,63] --> [1,40,63]
            first_point_enc_aug = first_point_enc_aug.permute(1,0,2)


        first_point_enc_aug = torch.cat([first_point_enc_aug, vel, latents], dim=-1)
        
        ## Concat Layer
        # for i, l in enumerate(self.concat_net):
        #     first_point_enc_aug = self.concat_net[i](first_point_enc_aug)
        #     if i < 2:
        #         first_point_enc_aug = F.relu(first_point_enc_aug)
        ####
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict1)
        sol_y = sol_y.squeeze()

        print("Sol y: ", sol_y.shape)

        # sol_y_x = sol_y[..., -512:]
        # sol_y_pose = sol_y[..., :8]
        
        pred_x = torch.squeeze(self.decoder(sol_y)) + static_latent
        print("Pred X: ", pred_x.shape)
        divergences = []
        for i in range(pred_x.shape[0]):
            
            div = torch.autograd.grad(outputs=pred_x[i], inputs=time_steps_to_predict, grad_outputs=torch.ones_like(pred_x[i]), create_graph=True, retain_graph=True)[0]
            print("Grad: ", div.shape)
            divergences.append(div)

        # for i in range(sol_y.shape[0]):
            
        #     div = torch.autograd.grad(outputs=sol_y[i], inputs=time_steps_to_predict, grad_outputs=torch.ones_like(sol_y[i]), create_graph=True, retain_graph=True)[0]
        #     print("Grad: ", div.shape)
        #     divergences.append(div)
        
        pred_pose = torch.squeeze(self.decoder_pose(sol_y))
        return torch.squeeze(pred_x), torch.stack(divergences), pred_pose
        # return sol_y.squeeze(), torch.stack(divergences), pred_pose


    def next_latent_noise(self, latent, times_obs, times_to_pred, angle=None, loc=None, run_backwards=True, n_traj_samples=1, vel=[0.,0.]):

        if not angle:
            angle=loc
        #truth_time_steps = torch.squeeze(times_obs)
        truth_time_steps = torch.tensor([0.], requires_grad=True)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        time_steps_to_predict = torch.squeeze(times_to_pred)
        time_steps_to_predict.requires_grad = True
        time_steps_to_predict1 = time_steps_to_predict.flip(0)
        
        angle_input = torch.tensor(angle)

        all_angle_input = []
        ## Add Noise to Noise
        noise = []
        for _ in range(30):
            angle_input = angle_input + torch.randn(angle_input.shape) * 0.1
            all_angle_input.append(angle_input)

        angle_input = torch.stack(all_angle_input)

        angle_input.requires_grad = True
        if latent==None:
            static_latent = self.static_latent_net(torch.tensor([0])).float()
            latents = self.latent_net(torch.tensor([0])).float()

            
            angle = self.embed_angle(angle_input).squeeze()
            # all_angle = []
            # ## Add Noise to Noise
            # noise = []
            # for _ in range(30):
            #     angle = angle + torch.randn(angle.shape) * 1
            #     all_angle.append(angle)

            # angle = torch.stack(all_angle)
            if len(angle.shape) == 1:  
                truth = angle.unsqueeze(0).unsqueeze(0)
            else:
                truth = angle.unsqueeze(1)

            truth_w_mask = truth
            print(truth_w_mask.shape)
            
            latents = latents.repeat(angle.shape[0], 1)
            if len(latents.shape) == 1: 
                latents = latents.unsqueeze(0).unsqueeze(0)
            else:
                latents = latents.unsqueeze(0)

        if self.z0_encoder_type == 'mlp':
            h = truth_w_mask
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                if i < 8:
                    h = F.relu(h)
                if i == 4:
                    h = torch.cat([truth_w_mask, h], -1)
                    
            first_point_enc_aug = h
            
            #print("First Apoint Enc Aug: ", first_point_enc_aug.shape)
            ### [40,1,63] --> [1,40,63]
            first_point_enc_aug = first_point_enc_aug.permute(1,0,2)


        #first_point_enc_aug = torch.cat([latents, first_point_enc_aug], dim=-1)
        vel = torch.tensor(vel)
        vel.requires_grad = True
        vel = self.embed_vel(vel)
        vel = vel.repeat(angle.shape[0], 1)
        if len(vel.shape) == 1: 
            vel = vel.unsqueeze(0).unsqueeze(0)
        else:
            vel = vel.unsqueeze(0)

        first_point_enc_aug = torch.cat([first_point_enc_aug, vel, latents], dim=-1)

        ## Concat Layer
        # for i, l in enumerate(self.concat_net):
        #     first_point_enc_aug = self.concat_net[i](first_point_enc_aug)
        #     if i < 2:
        #         first_point_enc_aug = F.relu(first_point_enc_aug)
        ####
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict1)


        # sol_y_x = sol_y[..., -512:]
        # sol_y_pose = sol_y[..., :8]
        
        pred_x = torch.squeeze(self.decoder(sol_y)) + static_latent
        print("Pred X: ", pred_x.shape)
        divergences = []
        for i in range(pred_x.shape[0]):
            
            div = torch.autograd.grad(outputs=pred_x[i], inputs=time_steps_to_predict, grad_outputs=torch.ones_like(pred_x[i]), create_graph=True, retain_graph=True)[0]
            print("Grad: ", div.shape)
            divergences.append(div)

        # sol_y = sol_y.squeeze()
        # for i in range(sol_y.shape[0]):
            
        #     div = torch.autograd.grad(outputs=sol_y[i], inputs=time_steps_to_predict, grad_outputs=torch.ones_like(sol_y[i]), create_graph=True, retain_graph=True)[0]
        #     print("Grad: ", div.shape)
        #     divergences.append(div)
        
        pred_pose = torch.squeeze(self.decoder_pose(sol_y))
        return torch.squeeze(pred_x), torch.stack(divergences), pred_pose
        # return sol_y.squeeze(), torch.stack(divergences), pred_pose