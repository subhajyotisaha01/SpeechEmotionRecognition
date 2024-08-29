# -*- coding:UTF-8 -*-
# Library import
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from Utils import get_all_data, get_all_dataset
from Models import CPAC_Model
import pandas as pd
from Config import Config
import parsing_file


# GPU configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

def main():

    parser = parsing_file.create_parser()
    args = parser.parse_args()


    # Paths and parameter settings
    # DATA_PATH = 'CASIA_88'
    # DATA_PATH ="/home/subhajyoti/CTL-MTNet/Dataset/EmoDB"
    # DATA_PATH = "/home/subhajyoti/CTL-MTNet/Dataset/SUBESCO"
    # DATA_PATH = "/home/subhajyoti/CTL-MTNet/Dataset/BanglaSER"
    # data_name = "emoDB"
    # data_name = "SUBESCO"
    # data_name = "BanglaSER"
    CLASS_LABELS = Config.CLASS_LABELS
    model_name = 'CPAC'
    save_model_name = 'CPAC'
    if_load = 1
    feature_method = 'mfcc'


    # k = 10
    # random_seed = 98
    repeat_number = 1

    # Read the data
    # x, y = get_all_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = False)
    # Custom Read Data for EmoDB
    x, y = get_all_dataset(args)

    # Create model
    y = to_categorical(y,num_classes=len(Config.CLASS_LABELS))
    args.digit_cap_num_capsule = len(Config.CLASS_LABELS) # Last Digit Cap Capsule Layer Dimension
    in_shape = x[0].shape
    x = x.reshape(x.shape[0], in_shape[0], in_shape[1])
    data_shape = x.shape[1:]
    model = CPAC_Model(input_shape = data_shape, num_classes = len(Config.CLASS_LABELS))

    # Train model
    print('Start Train CPAC in Single-Corpus')
    for number_times in range(0,repeat_number):
        i = 0
        data1 = []
        df = []
        model.matrix = []
        model.train(args, x, y, None, None, data_name = args.data_name, fold = args.k, random = args.random_seed + number_times)
        if args.result_distinguish == None:
            writer = pd.ExcelWriter("Results/"+args.data_name+'/'+str(args.k)+'fold_'+str(round(model.acc*10000)/100)+'_'+str(args.random_seed+ number_times)+'.xlsx')
        else:
            writer = pd.ExcelWriter("Results/"+args.data_name+'/'+args.result_distinguish+'_'+str(args.k)+'fold_'+str(round(model.acc*10000)/100)+'_'+str(args.random_seed+ number_times)+'.xlsx')
        for (i,item) in enumerate(model.matrix):
            temp = {}
            temp[" "] = CLASS_LABELS
            j = 0
            for j,l in enumerate(item):
                temp[CLASS_LABELS[j]]=item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer,sheet_name=str(i), encoding='utf8')

            df = pd.DataFrame(model.eva_matrix[i]).transpose()
            df.to_excel(writer,sheet_name=str(i)+"_evaluate", encoding='utf8')

        writer.save()
        writer.close()
        tf.keras.backend.clear_session()
        print('End of the '+str(number_times+1)+' training session')

    return ' '

if __name__ == "__main__":

    main()