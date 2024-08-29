import argparse

def create_parser():

    # You have to look at this parsing file and the configuration file to make the necessary changes.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',
                        default=0.1,
                        type=float,
                        help='test')
    parser.add_argument('--dataset_path', 
                        # default = "/home/subhajyoti/CTL-MTNet/Dataset/SUBESCO_fea_etract_acc_paper_1",
                        # default = "/home/subhajyoti/CTL-MTNet/Dataset/EmoDB",
                        default = "/home/subhajyoti/CTL-MTNet/Dataset/BanglaSER_MFCC_25_Fea_Extrac_Acc_Paer_1",
                        type = str,
                        help = "Trainning Dataset Path")
    parser.add_argument("--data_name",
                        default = "EmoDB",
                        type =str,
                        help = "Name of the Dataset being used")
    parser.add_argument("--k",
                        default = 10,
                        type = int,
                        help = "Use as 'K' for K-fold Cross Validation.")
    parser.add_argument("--result_distinguish",
                        # default = 'SUBESCO_using_weiner_filter_and_feature_extraction_from_paper_1',
                        default = "BanglaSER",
                        type = str,
                        help = "A string to recognize the eperiment details from the Reults directory.")
    parser.add_argument("--random_seed",
                        default = 98,
                        type = int,
                        help = "Random Seed")
    # parser.add_argument("--digit_cap_num_capsule",
    #                     default = 5,
    #                     type = int,
    #                     help = "Number of Capsules in the DigiCap Layer.")
    parser.add_argument("--digit_cap_capsule_dim",
                        default = 16,
                        type = int,
                        help = "Dimension of DigiCap Capsule.")
    parser.add_argument("--mse_weight",
                        default = 0.375,
                        # default = 0.0,
                        type = float,
                        help = "Co-efficient of the MSE loss between the predicted output and the smoothed target label.")
    parser.add_argument("--num_epochs",
                        default = 300,
                        type = int,
                        help = "The Number of Epochs")
    parser.add_argument("--margin_loss_weight",
                        default = 1.0,
                        type = float,
                        help = "Weight Co-efficient for Margin Loss Weight.")
    parser.add_argument("--if_reconstruction",
                        default = False,
                        type = bool,
                        help = "Whether the decoder part of the original capsule net would be added or not, i.e. whther \
                        we will use some fully connected layer to reconstruct the original image.")
    # parser.add_argument("--n_mfcc",
    #                     default = 39,
    #                     type = int,
    #                     help = "Number of MFCC Features to extract.")
    # parser.add_argument("--n_fft_sec",
    #                     default = None,
    #                     type = float,
    #                     help = "The window length in second.")
    # parser.add_argument("--n_fft",
    #                     default = None,
    #                     type = int,
    #                     help = "The window length in number of samples.")
    # parser.add_argument("--hop_length_sec",
    #                     default = None,
    #                     type = float,
    #                     help = "The Hop-Length in Seconds.")
    # parser.add_argument("--hop_length",
    #                     default = None,
    #                     type = int,
    #                     help = "The Hop-Length in number of samples.")
    parser.add_argument("--target_dataset_name",
                        default = "banglaser_data.npy",
                        # default = "emoDB_data.npy",
                        type = str,
                        help = "Target Dataset Name, a .npy File.")
    parser.add_argument("--target_dataset_label",
                        default = "banglaser_label.npy",
                        # default = "emoDB_label.npy",
                        type = str,
                        help = "Target Dataset Label Name a .npy File.")
    parser.add_argument("--capsule_drop_prob",
                        # default = 0.25,
                        default = None,
                        type = float,
                        help = "Capsule Drop Out Probability.")
    


    # Many more arguments

    return parser