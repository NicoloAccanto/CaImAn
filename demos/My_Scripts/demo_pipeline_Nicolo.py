import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.visualization import plot_contours

import caiman.fendo_functions_new as fendo

#%%
def main():
    pass  # For compatibility between running in an IDE and the CLI

    #%% Manually select file(s) to be processed. 
    
    main_path = '/Users/nicoloaccanto/Documents/Research/Projects/FiberBundle/Analysis/ExampleData'
    experiment_day = '2024_01_29'
    hour_type = '16_16_49_Photostimulation'
    
    data_dir = fendo.data_dir(main_path, experiment_day, hour_type)
    opts, Opt_res, dxy = data_dir.load_opt_param()
    
    # The data_dir functions sets all the folder paths.
    # opts, res, dxy = data_dir.load_opt_param() initializes parameters or 
    # loads them if they were previously saved in a previous analysis
    # calling opts.data you can see some of the previously saved parameters
    # If Opt_res = 0 and dxy =0, it means that it the 
    # first time that you run analysis on this files. opts.data is populated with
    # default values from caiman
    
    #%% Now we set some parameters. If it is the first time you analyze these data
    # you have to set Optical resolution, decay time of the Gcamp and the sigma for smoothing.
    # If the file was already analysed and you don't want to change these parameters
    # skip this cell.
    
    Opt_res = 0.87 # In µm and for binning 1. Depends on optical config. Change for GRIN to 0.62
    decay_time = 1.5 # depends on gcamp. It is expressed in seconds
    
    
    #%% Create a smoothed version of the file or load one, 
    #especially useful for getting rid of the fiber cores
    # If there is already a smoothed movie saved, this will overwrite it
    
    sigma = 0.8 #Change this for more or less smoothing
    original_movie, movie_smooth = data_dir.do_smoothing(sigma = sigma) #Import videos and create smoothed version
    
    #figure
    fig1, ax1 = plt.subplots(1,2, figsize = [8,8])
    ax1[0].imshow(original_movie[0,:,:])
    ax1[1].imshow(movie_smooth[0,:,:])
    
    #%% Show movie. Can skip this if you want
    try:
        m_orig = cm.load_movie_chain(data_dir.fname)
        m_smooth = cm.load_movie_chain(data_dir.fname_smooth)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio),
                                      m_smooth.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.9, magnification=2)  # press q to exit
    except AttributeError:
        m_orig = cm.load_movie_chain(data_dir.fname)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(fr=60, q_max=99.9, magnification=2)
    
    # %% Setup some parameters for motion correction
    # If you had already performed an analysis on this file, from now on you 
    # will overwrite all the parameter files and everything. So if you run this
    # then you should go to the end of the analysis

    # Decide which type of correction and on which movie to perform it. 
    use_smooth = True # Decide if smoothed or original movie for motion correction
    pw_rigid = False    # False to select rigid motion correction
    
    # You can also change the following parameters for better corrections, especially the first
    patch_motion_um = (40, 40)  # patch size for non-rigid correction in um
    max_shift_um = (10., 10.)       # maximum shift in um
    overlaps = (25,25)      # Overlap among patches
    max_deviation_rigid = 5 # maximum deviation allowed for patch with respect to rigid shifts
    
    fnames, dxy, max_shifts, strides = data_dir.format_optic_param(
        Opt_res, max_shift_um, patch_motion_um, use_smooth)
    
    mc_dict = {
        'fr': data_dir.Frame_rate,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy'
    }
    
    opts, basename = data_dir.save_opt_param(use_smooth, mc_dict)
    
    
   # %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)
    
    # %%% MOTION CORRECTION
    mc = fendo.MotCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    
    # %% compare with original movie
    m_orig = cm.load_movie_chain(fnames)
    m_els = cm.load(mc.mmap_file)
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0 
    ds_ratio = 0.2
    moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                  m_els.resize(1, 1, ds_ratio)], axis=2)
    moviehandle.play(fr=60, q_max=99.99, magnification=2)  # press q to exit
    
    #%% Analyse motion correction
    shifts_rig, shits_nr = mc.MotCorr_plots(data_dir.MC_path, basename, 
                                            data_dir.Frame_rate, dxy, min_shift =3)
    
    
    # %% MEMORY MAPPING 
    # memory map the file in order 'C' and delete the order 'F
    
    fname_new = cm.save_memmap(mc.mmap_file, base_name=basename, order='C',
                               border_to_0=border_to_0, dview=dview) # exclude borders
    os.remove(mc.mmap_file[0]) #This removes the order F file
    
    
    #%% Only reload the files you want to analyse. It is possible that you'll have 
    #   several saved, rigid non rigid, smoothed or not. Now it is the time to decide
    
    # You can restart the pipeline from here, but first you have to run the first
    # three cells.
    
    use_smooth = True # Decide if to use smoothed or not
    pw_rigid = False   # Decide if to use rigid or not
    
    mc_dict = {'pw_rigid': pw_rigid,}
    opts, basename = data_dir.save_opt_param(use_smooth, mc_dict)
    
    movie_chosen = glob.glob(data_dir.my_path + '/' + basename + '*.mmap')[0]
    Yr, dims, T = cm.load_memmap(movie_chosen)
    images = np.reshape(Yr.T, [T] + list(dims), order='F') 
    

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # %%  parameters for source extraction and deconvolution
    p = 1                    # order of the autoregressive system
    gnb = 2                  # number of global background components, better if 2, for paper I used 1
    merge_thr = 0.95         # merging threshold, max correlation allowed
    
    neuron_diam_um = 16      # diameter of a neuron in microns. Adjust this parameter if needed (but in general not needed)
    
    K = 4                    # number of components per patch
    
    gSig, stride_cnmf, rf_0 = data_dir.format_segmentation_parameters(neuron_diam_um,
                                                                    data_dir.dxy)
    
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'   #'sparse_nmf'#'greedy_roi'
    ssub = 1                     # spatial subsampling during initialization
    tsub = 1                     # temporal subsampling during intialization
    
    # parameters for component evaluation
    opts_dict = {'p': p,
                 'nb': gnb,
                 'merge_thr': merge_thr,
                 'stride': stride_cnmf,
                 'K': K,
                 'gSig': gSig,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'ssub': ssub,
                 'tsub': tsub}
    
    opts, basename = data_dir.save_opt_param(use_smooth, opts_dict)
    
    #%% Load or create templates from which to start detecting neurons
    
    # There are two options: either you start from the video in memory (already motion corrected)
    # and you start taking the mean or the std of the image and correct for the thresholds and stuff
    # or you start from a Ref_Image.tif that has to be in the same folder as the original file
    # If in the second plot there is an empty image, that means that the file Ref_Image.tif was 
    # not found. 
    
    use_Previous_ROI = False # If True, loads ROI from previous acquisition, and skips the reference image
    
    use_Ref_image = False # If True, loads the Ref image, if False takes the average of the video. 
                        # Only if use_Previous_ROI is False
    
    Cn = fendo.produce_template(np.std(images, axis=0), [0,0], 0, False)    
    
    if use_Previous_ROI:
        ROI = cnmf.load_CNMF(data_dir.my_path + 
                                      '/Ref_components.hdf5')
        a = ROI.estimates.A.toarray()
        a =  a > 0
        Ain = a
        if np.shape(a)[0] > np.shape(Cn)[0] * np.shape(Cn)[1]:
            b = np.zeros([np.shape(Cn)[0] * np.shape(Cn)[1], np.shape(a)[1]])
            for i in range(np.shape(a)[1]):
                a_map = np.reshape(a[:,i], [int(np.sqrt(np.shape(a)[0])),int(np.sqrt(np.shape(a)[0]))])
                b_map = fendo.rebin(a_map, [np.shape(Cn)[0], np.shape(Cn)[1]])
                b[:,i] = np.reshape(b_map, np.shape(b)[0])
            Ain = b
        Ain = Ain > 0
        crd = plot_contours(Ain, Cn)
        plt.title('Reloaded cells')
        
    else:    
        image_process, image_load = fendo.plot_template_image(
            images, data_dir.my_path, show_fig = True,
            sigma1 = [0,0], threshold1 = 0, sqrt1 = False, 
            sigma2 = [0,0], threshold2 = 0, sqrt2 = False)
    
        if use_Ref_image:
            mR = image_load 
        else:
            mR = image_process
    
        if np.mod(gSig[0],2) == 1:
            gSigseed = gSig[0]
        else:
            gSigseed = gSig[0] + 1
    
        fig1, ax1 =plt.subplots()
        Ain = cm.base.rois.extract_binary_masks_from_structural_channel(
            mR, gSig = gSigseed, expand_method='dilation')[0]
        crd = plot_contours(Ain.astype('float32'), Cn)
        plt.title('Segmented cells')[0]
        crd = plot_contours(Ain.astype('float32'), Cn)
    
    # # %% Other way to calculate average image
    # Cns = local_correlations_movie_offline(movie_chosen,
    #                                        remove_baseline=True, window=1000, stride=1000,
    #                                        winSize_baseline=100, quantil_min_baseline=10,
    #                                        dview=dview)
    # Cn = Cns.max(axis=0)
    # Cn[np.isnan(Cn)] = 0
    # cnm_template.estimates.plot_contours(img=Cn)
    # plt.title('Contour plots of found components');
    
    # %%  Decide if to start from the above found template
    # If you select False it is much slower, because it performs both the fit
    # and the refit in the same round. If it is too long we can separate them again
    
    start_from_template = True
    
    if not os.path.exists(data_dir.Imaging_path):
        os.makedirs(data_dir.Imaging_path)   
    
    if start_from_template:
        rf = None
        only_init = False
        opts_dict = {'rf': rf,'only_init': only_init}
        opts, basename = data_dir.save_opt_param(use_smooth, opts_dict)
        cnm_template = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
        cnm_template.fit(images)
        cnm_template.estimates.Cn = Cn
        cnm_template.save(data_dir.Imaging_path + '/ComponentsFromTemplate.hdf5')
    else: 
        rf = rf_0
        only_init = True
        opts_dict = {'rf': rf,'only_init': only_init}
        opts, basename = data_dir.save_opt_param(use_smooth, opts_dict)
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm.fit(images)
        cnm2 = cnm.refit(images, dview=dview)
        cnm2.estimates.Cn = Cn 
        cnm2.save(data_dir.Imaging_path + '/Components.hdf5')
    #%% Can restart from here if the process was stopped before. 
    # In case, re-run cells 1 and cell 'Only reload'
    
    # Load and plot results
    if os.path.exists(data_dir.Imaging_path + '/ComponentsFromTemplate.hdf5'):
        cnm_template = cnmf.load_CNMF(data_dir.Imaging_path + 
                                      '/ComponentsFromTemplate.hdf5')
        Cn = cnm_template.estimates.Cn
        cnm_template.estimates.plot_contours(img=Cn)
        
    if os.path.exists(data_dir.Imaging_path + '/Components.hdf5'):
        cnm2 = cnmf.load_CNMF(data_dir.Imaging_path + '/Components.hdf5')
        Cn = cnm2.estimates.Cn
        cnm2.estimates.plot_contours(img=Cn)  
        
    # %% COMPONENT EVALUATION and Plot
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 2  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected
    
    quality_dict = {'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest}
    
    if 'cnm_template' in locals():
        cnm_template.params.set('quality', quality_dict)
        cnm_template.estimates.evaluate_components(
            images, cnm_template.params, dview=dview)
        cnm_template.estimates.plot_contours(img=Cn,
                                          idx=cnm_template.estimates.idx_components)
        plt.suptitle('Cells found from TEMPLATE \
                     \n number of found components: {}'.format(
                     np.shape(cnm_template.estimates.idx_components)[0]))
        
        
    if 'cnm2' in locals():
        cnm2.params.set('quality', quality_dict)
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
        cnm2.estimates.plot_contours(img=Cn,
                                          idx=cnm2.estimates.idx_components)
        plt.suptitle('Cells found no template\
                      \n number of found components: {}'.format(
                      np.shape(cnm2.estimates.idx_components)[0]))

        
    opts, basename = data_dir.save_opt_param(use_smooth, quality_dict)


    #%% Decide which files to use
    use_template = True
    if use_template:
        cnm_final = cnm_template
    else:
        cnm_final = cnm2

# %% VIEW TRACES (accepted)
    cnm_final.estimates.view_components(img=Cn,
                                      idx=cnm_final.estimates.idx_components)


    #%% Extract DF/F values
    cnm_final.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    #%% Remove/restore components
    
    # First, check if we want to remove components out of the fiber or on the  borders
    
    tresh_dist = 500 # distance in pixels from the center
    
    index_out = fendo.find_index_components_out_border(
        cnm_final.estimates.A, cnm_final.estimates.idx_components, 
        dims[0], tresh_dist)
    
    
    
    to_remove = np.array([0]) # Chose elements from accepted components
                          #that you don't want, if none, then put np.array([0])
    to_restore = np.array([0]) # Chose elements from rejected components to restore,
                            #if non then use np.array([0])
    
    to_remove = np.unique(np.concatenate((index_out, to_remove))) 
    
    cnm_final.estimates.idx_components, cnm_final.estimates.idx_components_bad = \
        fendo.remove_restore_comp(to_remove, to_restore, cnm_final)

    cnm_final.estimates.plot_contours(
        img = Cn, idx=cnm_final.estimates.idx_components)


    #%% For photostim
    centerROI = cm.base.rois.com(
        cnm_final.estimates.A,dims[0],dims[1],1) #Center of mass of all the cells found
    
    centerROI_kept = cm.base.rois.com(
        cnm_final.estimates.A[:,cnm_final.estimates.idx_components],
                                 dims[0],dims[1],1) #Center of mass of all the cells found
    centerROI_rejected = cm.base.rois.com(
        cnm_final.estimates.A[:,cnm_final.estimates.idx_components_bad],
                                 dims[0],dims[1],1) #Center of mass of all the cells found
    
    coo_spots = data_dir.xy_photostim
    
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].imshow(Cn)
    ax1[0].scatter(centerROI_kept[:,1], centerROI_kept[:,0], color = 'r', marker = 'o', s = 5)
    ax1[0].scatter(coo_spots[:,0], coo_spots[:,1], color = 'y', marker = '*', s = 1)
    ax1[0].set_title('Kept components')
    
    ax1[1].imshow(Cn)
    ax1[1].scatter(centerROI_rejected[:,1],
                   centerROI_rejected[:,0], color = 'r', marker = 'o', s = 5)
    ax1[1].scatter(coo_spots[:,0], coo_spots[:,1],
                   color = 'y', marker = '*', s = 1)
    ax1[1].set_title('Rejected components')
    for i in range(len(centerROI_rejected)):
        x = centerROI_rejected[i,1]
        y = centerROI_rejected[i,0]
        ax1[1].text(x, y, i + 1, fontsize=8, color = 'w')

    

    #%% Select only high quality components
    
    cnm_final.estimates.select_components(use_object = True)
    cnm_final.save(data_dir.Imaging_path + 
                                      '/manually_inspected_components.hdf5')
    
    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    #%% Reload the file, initialize analysis object.
    # Can retake from here if the file manually_inspected_components already exists
    
    if os.path.exists(data_dir.Imaging_path + '/manually_inspected_components.hdf5'):
        cnm_final = cnmf.load_CNMF(data_dir.Imaging_path + 
                                      '/manually_inspected_components.hdf5')
        Cn = cnm_final.estimates.Cn
        cnm_final.estimates.plot_contours(img=Cn)
    
    data_analysis = fendo.components_analysis(cnm_final, data_dir)
    
    #%% Calculate SNR and replot
    SNR_thresh = [2] # Plot only cells with SNR> some value. You can decide more than one value
    ver_shift = 0.5 # Change this if you want the different curves to be more or less spaces
    smooth_tresh = 1  # Change this if <=2 no smoothing, if > 2 you can decide the smooth level
    
    neurons_SNR, coor_SNR = data_analysis.plot_SNR(SNR_thresh, ver_shift, smooth_tresh)
    
    #%% Export SNR and DeltaF/F traces (row, not smoothed)
    # You can change some part of the name, by default it is the hour 
    # of the file, but can change it in case
    
    name = data_dir.hour_type[: 9]
    
    data_analysis.export_traces(name)
    data_analysis.export_traces_SNR(name, SNR_thresh[0], neurons_SNR)
    data_analysis.export_SNRvalues(name)
        
    #%% Plot traces from the cells you want
    rows = 2
    cells1 = [0, 10, 5]
    cells2 = [3, 4, 8, 9]
    cells = [cells1, cells2]
    color_points = ['white','white']
    colors_lines = ['black', 'red']
    legend = ['Cells1', 'Cells2']
    title = 'My_figure'
    savefigure = True
    
    data_analysis.plot_FOV_traces(rows, cells, color_points, colors_lines, legend, 
                                  smooth_tresh, ver_shift, title, savefigure)

        
    #%% Photostim analysis
    #For the moment the possibility of having mulitple patterns in the same acquisition is not considered
    # Select carefully the following parameters, they are important
    
    flip = True #Try False if distances are not well calculated
    thresh_dist = 10 #threshold distance in µm, cells that are closer than this
                     #value to photostimulated spots are considered targeted
    time_period = 2 # seconds, both for the baseline and photostim, this can be changed
    Diff_STD = 2 # How many STD outside the noise to consider Photostimulated
    ver_shift = 0.5 # Change this if you want the different curves in plots to be more or less spaced
    
    
    # Function that performs the analysis. 
    data_analysis.do_photostim_analysis(flip, thresh_dist, time_period, 
                                        Diff_STD)
    
    # Save analysis and export some txt files.As before, you can change the name
    name = data_dir.hour_type[: 9]
    data_analysis.save(data_dir.Photostim_path + '/Photostim_analysis.hdf5', name)
    
    
    
    #%% This imports all the previous analysis
    a = fendo.load_photostim_analysis(data_dir.Photostim_path + '/Photostim_analysis.hdf5')
    print(a.keys())
    
    #%% To plot cell targered, responding etc
    data_analysis.plot_targeted_cells(save_figure = True)
    
    #%% To plot some photostim traces 
    data_analysis.plot_photostim_traces(ver_shift, smooth_tresh,save_figure = True)
    
    #%% To plot averaged photostim traces
    data_analysis.plot_photostim_traces(ver_shift, smooth_tresh,
                                        save_figure = True, average = True)
    
    
    #%% Plot graphs on lateral resolution of photostim
    
    data_analysis.plot_photostim_lateral_resolution(save_figure = True)


# %%
# This is to mask the differences between running this demo in an IDE
# versus from the CLI
if __name__ == "__main__":
    main()
