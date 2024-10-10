#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:11:00 2023

@author: nicolo


This is a series of fucntions that are important for the 2P-FENDO setup


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
from scipy import ndimage
import os
import pickle
import caiman as cm
from caiman.source_extraction.cnmf import params as params
from caiman.motion_correction import MotionCorrect
from scipy.signal import savgol_filter



class data_dir():
    """
    Class to read metadata and to generate all the folders and files that will be needed for the analysis
    
    """
    def __init__(self, main_path, experiment_day, hour_type, Opt_res):
        
        self.hour_type = hour_type
        self.my_path = main_path + '/' + experiment_day + '/' + hour_type
        self.fname = [self.my_path + '/' + hour_type + '.tif']  # filename to be processed
        self.imaging_param = self.my_path + '/' + hour_type + '_Image_Parameters.txt'
        self.data_imaging = pd.read_csv(self.imaging_param, delimiter = '\t', header = None)
        self.data_imaging.index = self.data_imaging.pop(0)
        self.Frame_rate = float(self.data_imaging.T['Frame rate'])
        self.Bin = float(self.data_imaging.T['Binning'])
        self.Opt_res = Opt_res
        self.dxy = [self.Opt_res*self.Bin, Opt_res*self.Bin]
        
        # Path to all folders we will need
        self.MC_path = self.my_path + '/Motion_correction'
        self.Imaging_path = self.my_path + '/Imaging_Analysis'
        self.Photostim_path = self.my_path + '/Photostim_Analysis'
        
        
        # Next we read the photositmulation coordinates if it is a Photostim acquisition. The corresponding file
        # is named Photostim_Coordinates.csv

        if 'Photostim' in hour_type:
            Photostim_coord_file = self.my_path + '/' + 'Photostim_Coordinates.csv'
            Photostim_coor =  pd.read_csv(Photostim_coord_file, delimiter = ',', header = 0)
            self.xy_photostim = np.transpose(np.array([Photostim_coor['X'], Photostim_coor['Y']]))
        else:
            self.xy_photostim = None
                
                
    def do_smoothing(self, sigma):
        """
        Creates a smoothed version of the video that was originally in the folder and saves it in the current folder 
        if save_smoothed = True, default is False

        Parameters
        ----------
            
        sigma : float
            The sigma value for gaussian smoothing, the higher the more the smoothing
            
        save_smoothed : bool, optional
            Set it to True to save a tif smoothed video. The default is False.

        Returns
        -------
        None

        """
        
        original_movie = tf.imread(self.fname)
        fnames = self.my_path + '/' + self.hour_type + '_smoothed_movie.tif' 
        if os.path.exists(fnames):
            movie_smooth = tf.imread(fnames)
            
        else:
            movie_smooth = ndimage.gaussian_filter(original_movie, sigma = [0, sigma, sigma])
            tf.imwrite(fnames, data = movie_smooth)    

        self.fname_smooth = [fnames]
        
        return original_movie, movie_smooth
            
    def format_optic_param(self, max_shift_um, patch_motion_um, use_smooth):
        """
        Formats the optical parameters in a way that caiman can read

        Parameters
        ----------

        max_shift_um : TYPE
            DESCRIPTION.
        patch_motion_um : TYPE
            DESCRIPTION.

        Returns
        -------
        dxy : TYPE
            DESCRIPTION.
        max_shifts : TYPE
            DESCRIPTION.
        strides : TYPE
            DESCRIPTION.

        """
        
        max_shifts = [int(a/b) for a, b in zip(max_shift_um, self.dxy)]
        strides = tuple([int(a/b) for a, b in zip(patch_motion_um, self.dxy)])
        
        fnames, basename =self.which_file_to_use(use_smooth)
        
        return fnames, self.dxy, max_shifts, strides
    
    def format_segmentation_parameters(self, neuron_diam_um, dxy):
        """
        Formats the parameters for source extraction and deconvolution
        in a way that caiman can read

        Parameters
        ----------
        neuron_diam_um : TYPE
            DESCRIPTION.
        dxy : TYPE
            DESCRIPTION.

        Returns
        -------
        gSig : TYPE
            DESCRIPTION.
        stride_cnmf : TYPE
            DESCRIPTION.
        rf : TYPE
            DESCRIPTION.

        """
        neuron_radius_pixels = int(neuron_diam_um/2/dxy[0])
        gSig = [neuron_radius_pixels, neuron_radius_pixels]    # expected half size of neurons in pixels.

        stride_cnmf = 2*neuron_radius_pixels          # amount of overlap between the patches in pixels (should be neuron diameter)
        rf = gSig[0] * 4          # half-size of the patches in pixels. Should be 3-4 times bigger than gSig
        
        return gSig, stride_cnmf, rf
        
    
    def which_file_to_use(self, use_smooth):
        """
        Decide which file to use based on the parameter use_smooth

        Parameters
        ----------
        use_smooth : TYPE
            DESCRIPTION.

        Returns
        -------
        fnames : 
            the name of the file to use
            
        basename : string
            a name to use as basename for saving later files
        

        """
        
        if use_smooth:
            fnames = self.fname_smooth
            basename = 'smoothed_movie'
        else:
            fnames = self.fname
            basename = 'original_movie'
        
        return fnames, basename
    
    
    def save_opt_param(self, use_smooth, mc_dict = dict()):
        """
        Updates optical parameters both in the caiman params.CNMFParams object
        and in the saved file opts_param.pkl. 
        
        If a opts_param.pkl file alrady exists, first it reads it, then it
        updates with the mc_dict values that are given by the user, or 
        to the caiman default if nothing is passed

        Parameters
        ----------
        use_smooth : bool
            True if using smoothed file
            
        mc_dict : TYPE, optional
            If nothing is passed then caiman will set it to its default.
            Otherwise it is a dictnory cointaing the parameters one want to
            update

        Returns
        -------
        opts : dict
            The new parameter dictonary from caiman.

        """
        
        opt_param_fpath = self.my_path + '/opts_param.pkl'
        fnames, basename = self.which_file_to_use(use_smooth)
        
        if os.path.exists(opt_param_fpath):
            with open(opt_param_fpath, 'rb') as f:
                opts_init = pickle.load(f)
            opts_load = params.CNMFParams()
        
            tot_dict = dict()
            for key in opts_init.keys():
                tot_dict = merge_two_dicts(tot_dict, opts_init[key])
            tot_dict = merge_two_dicts(tot_dict, {'fnames': fnames})
            opts_load = params.CNMFParams.change_params(opts_load, tot_dict)
            
              
        else:
            opts_load = params.CNMFParams(params_dict=mc_dict)
                
        opts = params.CNMFParams.change_params(opts_load, params_dict=mc_dict)
        
        if opts.motion['pw_rigid']:
            basename = 'NonRigid_' + basename
        else:
            basename = 'Rigid_' + basename
        with open(opt_param_fpath, 'wb') as f:
            pickle.dump(params.CNMFParams.to_dict(opts), f)
            
        return opts, basename
                

class MotCorrect(MotionCorrect):
    def __init__(self, fname, min_mov=None, dview=None, max_shifts=(6, 6), niter_rig=1, splits_rig=14, num_splits_to_process_rig=None,
                 strides=(96, 96), overlaps=(32, 32), splits_els=14, num_splits_to_process_els=None,
                 upsample_factor_grid=4, max_deviation_rigid=3, shifts_opencv=True, nonneg_movie=True, gSig_filt=None,
                 use_cuda=False, border_nan=True, pw_rigid=False, num_frames_split=80, var_name_hdf5='mov',is3D=False,
                 indices=(slice(None), slice(None))):
        super().__init__(fname, min_mov, dview, max_shifts, niter_rig, splits_rig, num_splits_to_process_rig,
                     strides, overlaps, splits_els, num_splits_to_process_els,
                     upsample_factor_grid, max_deviation_rigid, shifts_opencv, nonneg_movie, gSig_filt,
                     use_cuda, border_nan, pw_rigid, num_frames_split, var_name_hdf5,is3D,
                     indices)
        
    def MotCorr_plots(self, MC_path, basename, Frame_rate, dxy, min_shift = 1, save_tiff = True):
        """
        Plots graphs of rigid and non rigid motion and saves them together with 
        the corrected tif movie

        Parameters
        ----------
        starting_path : TYPE
            DESCRIPTION.
        basename : TYPE
            DESCRIPTION.
        Frame_rate : TYPE
            DESCRIPTION.
        dxy : TYPE
            DESCRIPTION.
        min_shift : TYPE, optional
            DESCRIPTION. The default is 1.
        save_tiff : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """

        self.MotCorr_folder = MC_path
        if not os.path.exists(self.MotCorr_folder):
            os.makedirs(self.MotCorr_folder)   
            
        m_corrected = cm.load(self.mmap_file)
        time_axis = np.arange(0,np.shape(m_corrected)[0])/Frame_rate
        
        if save_tiff:
            tf.imwrite(self.MotCorr_folder + '/' + basename + '.tif', m_corrected)
            
        allxshift_r = np.array(self.shifts_rig)[:,0] * dxy[0]
        allyshift_r = np.array(self.shifts_rig)[:,1] * dxy[0]
        all_shift_tot_r = np.sqrt(allxshift_r**2 + allyshift_r**2)
        df_rigid_shift = pd.DataFrame({'Rigid x shift (µm)': allxshift_r,
                                      'Rigid y shift (µm)': allyshift_r},
                                      index = time_axis)
        df_rigid_shift.index.name = 'Time_axis'
        df_rigid_shift.to_csv(self.MotCorr_folder + '/' + basename.split('_')[1] + 
                              '_Rigid_Shifts.txt', sep='\t', index = True)
        
        try:
            a = np.max(self.x_shifts_els,axis=0)
            xshift_patches = np.array(self.x_shifts_els) * dxy[0]
            yshift_patches = np.array(self.y_shifts_els) * dxy[0]
            column_xheaders = ['Patch ' + str(i +1) + ' x shift(µm)'for i in 
                               range(np.shape(xshift_patches)[1])]
            column_yheaders = ['Patch ' + str(i +1) + ' y shift(µm)'for i in 
                               range(np.shape(yshift_patches)[1])]
            df_xshift = pd.DataFrame(xshift_patches,
                                      index = time_axis,
                                      columns = column_xheaders)
            df_xshift.index.name = 'Time_axis'
            df_yshift = pd.DataFrame(yshift_patches,
                                      index = time_axis,
                                      columns = column_yheaders)
            df_yshift.index.name = 'Time_axis'
            df_xshift.to_csv(self.MotCorr_folder + '/' + basename.split('_')[1] + 
                             'NonRigid_Xshift.txt', sep='\t', index = True)
            df_yshift.to_csv(self.MotCorr_folder + '/' + basename.split('_')[1] + 
                             'NonRigid_Yshift.txt', sep='\t', index = True)
            allxshift_nr = np.mean(xshift_patches[:, a > min_shift], axis = 1)
            allyshift_nr = np.mean(yshift_patches[:, a > min_shift], axis = 1)
        except AttributeError:
            
            allxshift_nr = np.zeros(np.shape(self.shifts_rig)[0])
            allyshift_nr = np.zeros(np.shape(self.shifts_rig)[0])
            
        all_shift_tot_nr = np.sqrt(allxshift_nr**2 + allyshift_nr**2)
            
        # Create a figure
        fig, axs = plt.subplots(1,2)
        axs[0].plot(time_axis, allxshift_r, color = 'blue')
        axs[0].plot(time_axis, allyshift_r, color = 'red')
        axs[0].plot(time_axis, all_shift_tot_r, color = 'green')
        axs[0].legend(['Along x axis', 'Along y axis', 'Total'])
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Mean Dispalcement (µm)')
        axs[0].set_title('Rigid Dispalcement')
        
        axs[1].plot(time_axis, allxshift_nr, color = 'blue')
        axs[1].plot(time_axis, allyshift_nr, color = 'red')
        axs[1].plot(time_axis, all_shift_tot_nr, color = 'green')
        axs[1].legend(['Along x axis', 'Along y axis', 'Total'])
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Mean Dispalcement (µm)')
        axs[1].set_title('Non Rigid Dispalcement')  

        plt.savefig(self.MotCorr_folder + '/Figure_Movement.pdf')
        
        return all_shift_tot_r, all_shift_tot_nr
    

def plot_template_image(images, my_path, show_fig, sigma1, threshold1, sqrt1, sigma2, threshold2, sqrt2):
    mean_image = np.std(images, axis = 0)
    image_2 = produce_template(mean_image, sigma1, threshold1, sqrt1)
    
    try:
        load_ref = tf.imread(my_path + '/' + 'Ref_Image.tif')
    except FileNotFoundError: 
        load_ref = np.zeros(np.shape(mean_image))
        
    image_3 = produce_template(load_ref, sigma2, threshold2, sqrt2)
    
    if show_fig:
        fig, ax = plt.subplots(1, 2, figsize = [10,10])
        ax[0].imshow(image_2)
        ax[0].set_title('Processed image')
        ax[1].imshow(image_3)
        ax[1].set_title('Reference image loaded')
    
    return image_2, image_3

    
def produce_template(image, sigma, threshold, sqrt):
    
    image_1 = ndimage.gaussian_filter(image, sigma)
    image_1 = image_1 - threshold
    image_1 = image_1.clip(min=0)
    if sqrt:
        image_1 = np.sqrt(image_1)
        
    return image_1

def remove_restore_comp (to_remove, to_restore, cnm_object):
    
    good_components = cnm_object.estimates.idx_components
    bad_components = cnm_object.estimates.idx_components_bad
    
    if to_remove[0] !=0:
        rem = cnm_object.estimates.idx_components[to_remove -1]
        good_components = np.setdiff1d(cnm_object.estimates.idx_components, rem)
        bad_components = np.sort(np.append(bad_components, rem))
    if to_restore[0] !=0:
        rest = cnm_object.estimates.idx_components_bad[to_restore -1]
        good_components = np.sort(np.append(good_components, rest))
        bad_components = np.setdiff1d(cnm_object.estimates.idx_components_bad, rest)
    
    return good_components, bad_components

class components_analysis():
    
    def __init__(self, cnm_object, data_info):
        
        self.estimates = cnm_object.estimates
        self.data_info = data_info
        self.fr = data_info.Frame_rate
        self.time_axis = np.arange(0,np.shape(self.estimates.F_dff)[1])/self.fr
        self.Imaging_path = data_info.Imaging_path
        self.dxy = data_info.dxy[0]
        self.centerROI = cm.base.rois.com(self.estimates.A,
                                     cnm_object.dims[0],cnm_object.dims[1],1) #Center of mass of all the cells found
        
        if 'Photostim' in data_info.hour_type:
            self.xy_photostim = data_info.xy_photostim
            self.Photostim_path = data_info.my_path + '/Photostim_Analysis'
            
            phtostim_param_file = data_info.my_path + '/' + \
            data_info.hour_type + '_Photostim_Parameters.txt'
            
            data_photostim = pd.read_csv(phtostim_param_file, 
                                         delimiter = '\t', header = None)
            data_photostim.index = data_photostim.pop(0)
            self.Sec_1st_part = float(data_photostim.T['Sec 1st part'])
            self.Photostim_pulses = int(data_photostim.T['N Pulses'])
            self.Photostim_Rep = int(data_photostim.T['Repetitions'])
        

    def SNR_trace_plot(self, Nrows, neurons_to_plot, ver_shift,
                       smooth_tresh, SNR_thresh = [],
                       savefigure = True):
        
        ax_titles = [0]* Nrows
        
        data_smooth = self.data_smoothing(self.estimates.F_dff, smooth_tresh)
        
        fig, ax = plt.subplots(Nrows, 2, squeeze = False,
                               figsize = (10, 5*Nrows), sharex = 'col')
        
        for i in range(Nrows):

            ax[i,0].boxplot(self.estimates.SNR_comp)
            ax[i,0].set_title('Found {} components with SNR > {}'
                          .format(np.sum(self.estimates.SNR_comp>SNR_thresh[i]),
                                  SNR_thresh[i]))
            folder = self.Imaging_path
            fig_title = 'SNR_traces'
            ax_titles[i] = 'Traces of neurons with SNR > {}'.format(SNR_thresh[i])
            
            j=0
            for cell in neurons_to_plot[i]:
                ax[i,1].plot(self.time_axis, data_smooth[cell, :] + ver_shift*j,
                               linewidth = 0.8, color = 'black')
                j+=1
            ax[i,1].set_xlabel('Time (sec)')
            ax[i,1].set_ylabel('DF/F')
            ax[i,1].set_title(ax_titles[i])
        
        if savefigure:
            plt.savefig(folder + '/' + fig_title + '.pdf')
        
    def data_smoothing(self, traces_to_smooth, smooth_tresh):
        if smooth_tresh <= 2:
            data_smooth = traces_to_smooth
        else:
            data_smooth = savgol_filter(traces_to_smooth, smooth_tresh, 2)
            
        return(data_smooth)
        

    def plot_SNR (self, SNR_thresh, ver_shift, smooth_tresh):
        
        neurons_to_plot = [0]* len(SNR_thresh) 
        coor_neurons_to_plot = [0]* len(SNR_thresh) 
        for i in range(len(SNR_thresh)):
            neurons_to_plot[i], coor_neurons_to_plot[i] = self.chose_neurons_SNR(SNR_thresh[i])
            self.estimates.plot_contours(img = self.estimates.Cn, idx = neurons_to_plot[i])
            fig = plt.gcf()
            ax1, ax2 = fig.get_axes()
            ax1.set_title('Components with SNR > {}'.format(SNR_thresh[i]))
            ax2.set_title('Components with SNR < {}'.format(SNR_thresh[i]))
            plt.savefig(self.Imaging_path +'/Image_components_SNR_{}.pdf'.format(SNR_thresh[i]))
        
        Nrows = len(SNR_thresh)
        self.SNR_trace_plot(Nrows, neurons_to_plot, ver_shift,
                           smooth_tresh, SNR_thresh = SNR_thresh)
        
        return neurons_to_plot, coor_neurons_to_plot
    
    def chose_neurons_SNR(self, SNR_thresh):
        all_cells = np.arange(np.shape(self.estimates.F_dff)[0])  
        neurons_chosen = all_cells[self.estimates.SNR_comp > SNR_thresh]
        coo_neurons = self.centerROI[neurons_chosen, :2]
        
        return(neurons_chosen, coo_neurons)
    
    
    def photostim_param(self):
        photostim_duration = self.Photostim_pulses / self.fr
        deltaT_frames = self.Sec_1st_part*self.fr + 2*self.Photostim_pulses 
        # frames between two consecutive photstimualtions
        detltaT = deltaT_frames / self.fr
    
        self.photostim_end = np.array([(photostim_duration + self.Sec_1st_part) + (detltaT)* i 
                                  for i in range(self.Photostim_Rep)])
        self.photostim_start = self.photostim_end - photostim_duration
        
        return(photostim_duration, deltaT_frames, detltaT,
               self.photostim_end, self.photostim_start)
    
    def cell_distances(self, flip, thresh_dist):
        all_cells = np.arange(np.shape(self.estimates.F_dff)[0])

        coo_spots = self.xy_photostim
        if flip:
            coo_spots = np.flip(coo_spots,axis =1)
        distances = [np.sqrt(np.sum(np.power(self.centerROI[:, 0:2] * self.dxy 
                                     - coo_spots[i, :] * self.dxy, 2),
                            axis = 1)) for i in range(len(coo_spots))]

        distance_stim = np.min(distances, axis = 0)
        cells_targeted = all_cells[distance_stim < thresh_dist]
        coo_cells_targeted = self.centerROI[cells_targeted, :2] 
        cells_non_targeted = np.setdiff1d(all_cells, cells_targeted)
        
        return(self.centerROI, coo_spots, distance_stim, cells_targeted, coo_cells_targeted, cells_non_targeted)
        
    def add_zeros_to_end(self, deltaT_frames, all_cells):
        # Frames to remove
        frames_one_cycle = int(deltaT_frames)
        self.remove_frames = - (frames_one_cycle * self.Photostim_Rep) + \
        np.shape(self.estimates.F_dff)[1]
        
        # Add a bunch of zeros at the end of the DF/F files
        DF_start = np.delete(self.estimates.F_dff, 
                             range(int(np.floor(self.remove_frames/2))), axis = 1)
        DF_end = np.delete(DF_start, range(
            np.shape(DF_start)[1]-(int(np.floor(self.remove_frames/2))),
            np.shape(DF_start)[1]), axis = 1)
        DF_short = np.array(np.split(DF_end, self.Photostim_Rep, axis=1))
        
        DF_average = np.zeros([len(all_cells), frames_one_cycle])
        

        a = np.sum(DF_short[: self.Photostim_Rep, :, :],
                   axis = 0) / self.Photostim_Rep
        DF_average[:, : frames_one_cycle] = a
        
        return(DF_average)
    
    def is_photostim(self, time_period_frames, Diff_STD, all_cells):
        N_patterns=1
        
        photostim_start_frames = self.photostim_start[
            : N_patterns] * self.fr - int(np.floor(self.remove_frames/2))
        photostim_end_frames = self.photostim_end[
            : N_patterns] * self.fr - int(np.floor(self.remove_frames/2))
        
        baseline_std = np.zeros([len(all_cells), N_patterns])
        Signal_photostim = np.zeros([len(all_cells), N_patterns])
        is_cell_photostim = np.zeros([len(all_cells), N_patterns])
        
        for i in range(N_patterns):
            baseline_frames = np.arange(photostim_start_frames[i] - time_period_frames, 
                                    photostim_start_frames[i], dtype=int)#
            post_photostim_frames = np.arange(photostim_end_frames[i] , 
                                       photostim_end_frames[i] + time_period_frames, dtype=int)
            
            baseline_std[:, i] = np.std(self.DF_average[:, baseline_frames], axis = 1)
            Signal_photostim[:, i] = (np.average(self.DF_average[:, post_photostim_frames], axis = 1) - 
                                np.average(self.DF_average[:, baseline_frames], axis = 1))
            is_cell_photostim = Signal_photostim > Diff_STD * baseline_std
            
        return(Signal_photostim, is_cell_photostim)
    
    def targeted_responding(self, all_cells, is_cell_photostim, 
                            cells_targeted, cells_non_targeted):
        """
        

        Parameters
        ----------
        all_cells : TYPE
            DESCRIPTION.
        is_cell_photostim : TYPE
            DESCRIPTION.
        cells_targeted : TYPE
            DESCRIPTION.
        cells_non_targeted : TYPE
            DESCRIPTION.

        Returns
        -------
        cells_T_R : int
            cells targeted responding
        cells_NT_R : int
            cells non targeted responding
         cells_NT_NR : int
             cells non targeted non responding

        """
        cells_phostimulated = all_cells[is_cell_photostim[:,0]]
        mask = np.in1d(cells_phostimulated, cells_targeted)
        cells_T_R = cells_phostimulated[mask]

        
        cells_T_NR = np.setdiff1d(cells_targeted, cells_T_R)
        cells_NT_R = np.setdiff1d(cells_phostimulated, cells_T_R)
        cells_NT_NR = np.setdiff1d(cells_non_targeted, cells_NT_R) 
        
        return(cells_T_R, cells_T_NR, cells_NT_R, cells_NT_NR)
    
    
    
    def do_photostim_analysis (self, flip, thresh_dist, time_period, Diff_STD):
        
        if not os.path.exists(self.data_info.Photostim_path):
            os.makedirs(self.data_info.Photostim_path)
            
        photostim_duration, deltaT_frames, detltaT, \
            self.photostim_end, self.photostim_start = self.photostim_param()
        
        self.centerROI, self.coo_spots, self.distance_stim, self.cells_targeted, self.coo_cells_targeted, \
            self.cells_non_targeted = self.cell_distances(flip, thresh_dist)
            
        all_cells = np.arange(np.shape(self.estimates.F_dff)[0])
        self.DF_average = self.add_zeros_to_end(deltaT_frames, all_cells)
        
        time_period_frames = int(np.ceil(time_period*self.fr))
        
        self.Signal_photostim, self.is_cell_photostim = self.is_photostim(
            time_period_frames, Diff_STD, all_cells)
        
        self.cells_T_R, self.cells_T_NR, self.cells_NT_R, self.cells_NT_NR  = \
            self.targeted_responding(all_cells, self.is_cell_photostim, 
                                     self.cells_targeted, self.cells_non_targeted)
        
        self.coo_T_R = self.centerROI[self.cells_T_R, :2]
        self.coo_T_NR = self.centerROI[self.cells_T_NR, :2]
        self.coo_NT_R = self.centerROI[self.cells_NT_R, :2] 
        self.coo_NT_NR = self.centerROI[self.cells_NT_NR, :2] 
        
        photostim_dict = {
            'flip': flip,
            'thresh_distance': thresh_dist,
            'time_period': time_period,
            'DIfference_STD': Diff_STD}
        
        self.photostim_param_fpath = self.Photostim_path + '/Photostim_param.pkl'
        with open(self.photostim_param_fpath, 'wb') as f:
            pickle.dump(photostim_dict, f)
    
        
    def plot_targeted_cells(self, save_figure = True):   
        fig, axs = plt.subplots(1,2, figsize = (10,8))
        axs[0].imshow(self.estimates.Cn)
        a = axs[0].scatter(self.centerROI[:,1], self.centerROI[:,0], color = 'w', marker = 'o', s = 5)
        b = axs[0].scatter(self.coo_spots[:,1], self.coo_spots[:,0], color = 'r', marker = '*', s = 5)
        c = axs[0].scatter(self.coo_cells_targeted[:,1], self.coo_cells_targeted[:,0], 
                    facecolors='none', edgecolors='y', marker = 'o', s = 50)
        axs[0].legend([a,b,c],['found cells', 'photostim spots', 'cell considered targeted'],
                      fontsize = 8, bbox_to_anchor=(0.8, 1.2), facecolor = 'gray')
        for i in range(len(self.centerROI)):
            x = self.centerROI[i,1]
            y = self.centerROI[i,0]
            axs[0].text(x, y, np.floor(self.distance_stim[i]), fontsize=8, color = 'w')
            
        axs[1].imshow(self.estimates.Cn)
        a = axs[1].scatter(self.coo_T_R[:,1], self.coo_T_R[:,0], 
                           color = 'w', marker = 'o', s = 10)
        b = axs[1].scatter(self.coo_T_NR[:,1], self.coo_T_NR[:,0], 
                    facecolors='none', edgecolors='w', marker = 'o', s = 50)
        
        c = axs[1].scatter(self.coo_NT_R[:,1], self.coo_NT_R[:,0], color = 'r', marker = '*', s = 10)
        d = axs[1].scatter(self.coo_NT_NR[:,1], self.coo_NT_NR[:,0], color = 'blue', marker = '*', s = 10)
        axs[1].legend([a,b,c,d],['targeted responding', 'targeted non responding',
                                 'non targ resp', 'non targ non resp'],
                      fontsize = 8, bbox_to_anchor=(0.8, 1.22), facecolor = 'gray')
        
        if save_figure:
            plt.savefig(self.Photostim_path + '/Targeted and photostimulated cells.pdf')
            
    def plot_photostim_traces(self, ver_shift, smooth_tresh, 
                              save_figure = True, average = False):
        
        if average:
            fig1, ax1 = plt.subplots(2,2, figsize=(8,10), 
                                     gridspec_kw={'width_ratios': [2, 1]})  
        else:
            fig1, ax1 = plt.subplots(2,2, figsize=(10,10))
        
        color1 = ['w','r']
        symbol1 = ['o', 'o']
        legend1 = ['targeted responding', 'targeted non responding']
        self.show_image_coor(ax1[0,0], self.coo_T_R, self.coo_T_NR, color1, symbol1, legend1)
        
        color2 = ['r','b']
        symbol2 = ['*', '*']
        legend2 = ['non targeted responding', 'non targeted non responding']
        self.show_image_coor(ax1[1,0], self.coo_NT_R, self.coo_NT_NR, color2, symbol2, legend2)
        
        
        self.plot_traces(ax1[0,1], ver_shift, smooth_tresh,
                         self.cells_T_R, self.cells_T_NR, ['k', 'r'], average)  
        self.plot_traces(ax1[1,1], ver_shift, smooth_tresh,
                         self.cells_NT_R, self.cells_NT_NR, ['r', 'b'], average)  
        
        if save_figure:
            if average:
                plt.savefig(self.Photostim_path + '/Average Traces cells.pdf')
            else:
                plt.savefig(self.Photostim_path + '/Traces cells.pdf')
                
        

    def show_image_coor(self, axis, coord1, coord2, colors, symbol, legend):
        axis.imshow(self.estimates.Cn)
        axis.axis('off')
        
        a = axis.scatter(coord1[:,1], coord1[:,0], 
                            color = colors[0], marker = symbol[0], s = 10)
        b = axis.scatter(coord2[:,1], coord2[:,0], 
                    facecolors='none', edgecolors=colors[1], marker = symbol[1], s = 50)
        axis.legend([a,b],legend, 
                        fontsize = 8, bbox_to_anchor=(0.9, 1.16), facecolor = 'gray')
        
        self.cell_number_in_FOV(coord1, axis, color = colors[0])
        self.cell_number_in_FOV(coord2, axis, color = colors[1])
    
    
    def cell_number_in_FOV(self, coordinates, axis, color):
        for i in range(len(coordinates)):
            x = coordinates[i,1]
            y = coordinates[i,0]
            axis.text(x, y, i+1, fontsize=8, color = color)
            
    def plot_traces(self, axis, ver_shift, smooth_tresh, cells1, cells2, colors, average):
        
        if average:
            time_axis_reduced = self.time_axis[0:np.shape(self.DF_average)[1]]
            axis.plot(time_axis_reduced, np.transpose(self.DF_average[cells1, :]) + 
                      ver_shift * np.arange(len(cells1)), 
                    color = colors[0], linewidth = 0.5)
            
            axis.plot(time_axis_reduced, np.transpose(self.DF_average[cells2, :]) + 
                      ver_shift * (np.arange(len(cells2))+len(cells1)), 
                      color = colors[1], linewidth = 0.5)
            
            axis.vlines([self.photostim_start[:1]- int(np.floor(self.remove_frames/2)/self.fr),
                             self.photostim_end[:1] - int(np.floor(self.remove_frames/2)/self.fr)],
                            0, ver_shift * (len(cells2)+len(cells1)) - ver_shift/2, 
                            color = 'red')
            
            axis.set_xlabel('Time (sec)')
            
        else:
  
            data_smooth = self.data_smoothing(self.estimates.F_dff, smooth_tresh)
            axis.plot(self.time_axis, np.transpose(data_smooth[cells1, :]) + 
                      ver_shift * np.arange(len(cells1)), 
                    color = colors[0], linewidth = 0.5)
            
            axis.plot(self.time_axis, np.transpose(data_smooth[cells2, :]) + 
                      ver_shift * (np.arange(len(cells2))+len(cells1)), 
                      color = colors[1], linewidth = 0.5)
            
            axis.vlines([self.photostim_start[: self.Photostim_Rep],
                             self.photostim_end[: self.Photostim_Rep]],
                            0, ver_shift * (len(cells2)+len(cells1)) - ver_shift/2, color = 'red')
      
            axis.set_xlabel('Time (sec)')
        
    def plot_FOV_traces(self, rows, cells, colors_points, colors_lines, legend, 
                        smooth_tresh,ver_shift, title, savefigure):
        """
        This function generates a plot with a FOV on the right side, in which the cells passed
        to the function as a parameter appear, and on the righ twe have the traces of the 
        selected cells

        Parameters
        ----------
        rows : TYPE
            DESCRIPTION.
        cells : TYPE
            DESCRIPTION.
        colors : TYPE
            DESCRIPTION.
        legend : TYPE
            DESCRIPTION.
        smooth_tresh : TYPE
            DESCRIPTION.
        ver_shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        data_smooth = self.data_smoothing(self.estimates.F_dff, smooth_tresh)
        fig, ax = plt.subplots(rows, 2, squeeze=False)
        handlers = [0] *len(cells)
        
        for i in range(rows):
            ax[i,0].imshow(self.estimates.Cn)
            ax[i,0].axis('off')
            handlers[i] = self.show_coor_in_image(ax[i,0],cells[i], colors_points[i])
            ax[i,0].legend([handlers[i]],[legend[i]], fontsize = 8, 
                           bbox_to_anchor=(0.9, 1.16), facecolor = 'gray')
            
            ax[i,1].plot(self.time_axis, np.transpose(data_smooth[cells[i], :]) + 
                      ver_shift * np.arange(len(cells[i])), 
                    color = colors_lines[i], linewidth = 0.5)
        ax[-1,1].set_xlabel('Time (sec)')
        
        if savefigure:
            plt.savefig(self.Imaging_path + '/' + title + '.pdf')
            

    def show_coor_in_image(self, axis, cells, colors):
        cell_coo = self.centerROI[cells, :2]
        handlers = axis.scatter(cell_coo[:,1], cell_coo[:,0], facecolor = 'none',
                            edgecolors = colors, marker = 'o', s = 15)
        
        self.cell_number_in_FOV(cell_coo, axis, color = colors)
 
        return(handlers)
    
    
    def plot_photostim_lateral_resolution(self, save_figure):
        
        fig3, ax3 = plt.subplots(1,2, figsize = [8, 4])
        
        ax3[0].scatter(self.distance_stim, 
                               self.Signal_photostim)
        ax3[1].scatter(self.distance_stim, 
                              self.is_cell_photostim)
       
        ax3[0].set_xlabel('Distance from photostim cell (µm)')
        ax3[0].set_ylabel('Signal icnrease after Photostim')  
        
        ax3[1].set_xlabel('Distance from photostim cell (µm)')
        ax3[1].set_ylabel('Cell Photostimualted?')
        
        if save_figure:
            plt.savefig(self.Photostim_path + '/Lateral_resolution.pdf')
        
        
        
        
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of yß
    return z


