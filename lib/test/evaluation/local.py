from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data2/wangyujie/ODTrack/data/got10k_lmdb'
    settings.got10k_path = '/data2/wangyujie/ODTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data2/wangyujie/ODTrack/data/itb'
    settings.lasot_extension_subset_path = '/data2/wangyujie/ODTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data2/wangyujie/ODTrack/data/lasot_lmdb'
    settings.lasot_path = '/data2/wangyujie/ODTrack/data/lasot'
    settings.network_path = '/data2/wangyujie/ODTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data2/wangyujie/ODTrack/data/nfs'
    settings.otb_path = '/data2/wangyujie/ODTrack/data/otb'
    settings.prj_dir = '/data2/wangyujie/ODTrack'
    settings.result_plot_path = '/data2/wangyujie/ODTrack/test/result_plots'
    settings.results_path = '/data2/wangyujie/ODTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data2/wangyujie/ODTrack'
    settings.segmentation_path = '/data2/wangyujie/ODTrack/test/segmentation_results'
    settings.tc128_path = '/data2/wangyujie/ODTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data2/wangyujie/ODTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data2/wangyujie/ODTrack/data/trackingnet'
    settings.uav_path = '/data2/wangyujie/ODTrack/data/uav'
    settings.vasttarck_path = '/data2/wangyujie/ODTrack/data/vasttrack'
    settings.vot18_path = '/data2/wangyujie/ODTrack/data/vot2018'
    settings.vot22_path = '/data2/wangyujie/ODTrack/data/vot2022'
    settings.vot_path = '/data2/wangyujie/ODTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

