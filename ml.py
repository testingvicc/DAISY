#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4ml

###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import os
import glob
import re
import time
import datetime
import inspect

import json
import pickle

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

# tensorflow-keras (for GPU use)
# BUG FIX: Requires modifying the keras/backend/tensorflow_backend.py file:
# From:
#       return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)
# To:
#       return isinstance(x, tf_ops.core_tf_types.Tensor) or tf_ops.is_dense_tensor_like(x)
#
from tensorflow.keras.models            import Model, Sequential, load_model
from tensorflow.keras.layers            import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks         import History 

# vanilla keras
# from keras.models            import Model, Sequential, load_model
# from keras.layers            import Input, Dense, LSTM, TimeDistributed, RepeatVector
# from keras.callbacks         import History 

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.utils  as d4utl
import ds4n6_lib.evtx  as d4evtx
import ds4n6_lib.flist as d4flst


###############################################################################
# FUNCTIONS
###############################################################################


# POTENTIALLY NON-PUBLIC FUNCTIONS ############################################
# TODO: This is useful but we need to decide if it makes sense to include it
#       in the public distribution

def find_anomalies(indf, model_type="simple_autoencoder", **kwargs):
    if 'D4_DataType_' in indf.columns:
        if indf['D4_DataType_'][0]=='evtx':
            hml_df = d4evtx.find_anomalies_evtx(indf)
        elif indf['D4_DataType_'][0]=='flist':
            hml_df = d4flst.find_anomalies_flist(indf)
        elif indf['D4_DataType_'][0]=='evtx-hml' or indf['D4_DataType_'][0]=='flist-hml':
            d4evtx.find_anomalies_evtx(indf)
        else:
            print("DataFrame type not supported")
            return None, None
    else:
        print("DataFrame type not supported")
        return None, None
    
    hml_df = hml_df.drop(columns=['D4_DataType_',])

    if model_type == "simple_autoencoder":
        topanomdf, anomdf, lossdf  = ml_model_execution_quick_case(traindf=hml_df, preddf=hml_df, model_type=model_type, **kwargs)
        return anomdf, lossdf
    elif model_type == "lstm_autoencoder":
        topanomdf, anomdf, lossdf, lossftdf = ml_model_execution_quick_case(traindf=hml_df, preddf=hml_df, model_type=model_type, **kwargs)
        return anomdf, lossdf

def ml_model_execution_quick_case(**kwargs):

    # traindf                = kwargs.get('traindf',                None)
    # trainarr               = kwargs.get('trainarr',               None)
    # preddf                 = kwargs.get('preddf',                 None)
    # model_type             = kwargs.get('model_type',             None)
    # lstm_units             = kwargs.get('lstm_units',             None)
    # lstm_time_steps        = kwargs.get('lstm_time_steps',        None)
    # activation_function    = kwargs.get('activation_function',    None)
    # model_filename         = kwargs.get('model_filename',         None)
    # loops                  = kwargs.get('loops',                  1)
    # epochss                = kwargs.get('epochss',                [2])
    # error_ntop             = kwargs.get('error_ntop',             5000)
    # verbose                = kwargs.get('verbose',                0)
    # cols2drop              = kwargs.get('cols2drop',              [])
    # transform_method       = kwargs.get('transform:_method',      "label_encoder")
    # batch_size             = kwargs.get('batch_size',             8)
    # ntop_anom              = kwargs.get('ntop_anom',              500)
    # autosave_miloss_model  = kwargs.get('autosave_minloss_model', True)
    # maxcnt                 = kwargs.get('maxcnt',                 1)
    # activation_function    = kwargs.get('activation_function',    "tanh")
    
    return model_execution(**kwargs)

def model_execution(**kwargs):

    verbose                = kwargs.get('verbose',                0)
    traindf                = kwargs.get('traindf',                None)
    predindf               = kwargs.get('preddf',                 None)
    evilquerystring        = kwargs.get('evilquerystring',        None)
    cols2drop              = kwargs.get('cols2drop',              [])
    model_type             = kwargs.get('model_type',             "simple_autoencoder")
    epochss                = kwargs.get('epochss',                [10])
    ntop_anom              = kwargs.get('ntop_anom',               200)
    maxcnt                 = kwargs.get('maxcnt',                 1)
    # unused parameters
    # model_filename_root    = kwargs.get('model_filename_root',    None)
    # model_filename         = kwargs.get('model_filename',         None)
    evilqueryfield         = kwargs.get('evilqueryfield',         None)
    # loops                  = kwargs.get('loops',                  1)
    # batch_size             = kwargs.get('batch_size',             8)
    # error_ntop             = kwargs.get('error_ntop',             None)
    # autosave_minloss_model = kwargs.get('autosave_minloss_model', False)

    # Model-specific Arguments
    lstm_time_steps        = kwargs.get('lstm_time_steps',        200)

    if d4.debug != 0:
        verbose = d4.debug

    # Drop columns - During the Model Definition we can try dropping columns see if that improves our detection
    if traindf is not None:
        traindf = traindf.drop(columns=cols2drop)
    if predindf   is not None:
        predindf  = predindf.drop(columns=cols2drop)

    for epochs in epochss:
        display(Markdown("----"))
        display(Markdown("**"+str(epochs)+" epochs**"))
        evilentryidxs = []
        cnt=1

        while cnt <= maxcnt:
            if model_type == "lstm_autoencoder":
                # In the LSTM case we are also interested in the per-feature loss,
                # since the global loss is calculated as the mean of the losses of
                # the features
                if d4.debug == 5:
                     predinarr, predoutarr, anomdf, loss, lossft = ml_autoencoder(**kwargs, epochs=epochs)
                else:
                     anomdf, loss, lossft = ml_autoencoder(**kwargs, epochs=epochs)

                # We need to shift the predindf by lstm_time_steps-1 
                predinshifteddf = predindf.iloc[lstm_time_steps-1:]
                predinshifteddf.index -= lstm_time_steps-1

                if d4.debug >= 3:
                    display(predinshifteddf.head(3))
                    display(predinshifteddf.tail(3))

                # Sort anomdf by loss
                erroranomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index']).values)
                anomsorteddf = predinshifteddf.loc[erroranomidx].reset_index().rename(columns={"index": "Orig_Index"})
                if d4.debug >= 3:
                    print("DEBUG: erroranomidx -> "+str(type(erroranomidx))+str(len(erroranomidx)))
                    print("DEBUG: predinshifteddf -> "+str(type(predinshifteddf))+" -> "+str(predindf.iloc[lstm_time_steps-1:].shape))
                    print("DEBUG: anomsorteddf -> "+str(type(anomsorteddf))+str(anomsorteddf.shape))
                    display(anomsorteddf.head(3))
                    display(anomsorteddf.tail(3))
            else:
                anomdf, loss  = ml_autoencoder(**kwargs, epochs=epochs)

                # Sort anomdf by loss
                erroranomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index']).values)
                anomsorteddf = predindf.loc[erroranomidx].reset_index().rename(columns={"index": "Orig_Index"})

            if d4.debug >= 3:
                print("")
                print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
                print("DEBUG: anomdf.dtypes: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                print(anomdf.dtypes)
                print("DEBUG: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                print("")


            #erroranomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index']).values)
            #anomsorteddf = predindf.loc[erroranomidx].reset_index().rename(columns={"index": "Orig_Index"})

            # Find & Print our evil entries
            if (evilqueryfield is not None) and (evilquerystring is not None):
                query         = eval("evilqueryfield")+'.str.contains("'+evilquerystring+'")'
                evilentriesdf = anomsorteddf.query(query, engine="python")

                print("")
                display(Markdown("**Evil Entries:**"))
                display(evilentriesdf)
                print("")

                if evilentriesdf.shape[0] == 0:
                    print("ERROR: Evil Entry not found. Verify Injection.")
                    print("")
                else:
                    evilentryorigidx = evilentriesdf.head(1).index

                    if model_type == "lstm_autoencoder":
                        # The LSTM model drops the first time_steps values from the data,
                        # so the index needs to be corrected
                        evilentrylstmidx = evilentryorigidx 
                        evilentryidx = anomsorteddf.query('index == @evilentrylstmidx').index.values[0]
                    else:
                        evilentryidx = anomsorteddf.query('index == @evilentryorigidx').index.values[0]

                    # Save in list (multiple cnt-runs)
                    evilentryidxs.append(evilentryidx)

                    display(Markdown("**Best Evil Entry Error Order: "+str(evilentryidxs)+"**"))
                    print("")

            # Print Top Anomalies
            if ntop_anom > anomdf.shape[0]:
                ntop_anom = anomdf.shape[0]

            errortopanomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index'].head(ntop_anom)).values)
            if model_type == "lstm_autoencoder":
                # The LSTM model drops the first time_steps values from the data,
                # so the index needs to be corrected
                errortopanomidx = [x + lstm_time_steps - 1 for x in errortopanomidx]

            if d4.debug >= 3:
                print("DEBUG: [DBG"+str(d4.debug)+"] errortopanomidx: ", end='')
                print(errortopanomidx)

            topanomdf = anomdf.loc[errortopanomidx].reset_index()
            losssr    = pd.Series(loss)

            #anomdf = anomdf.loc[errortopanomidx].reset_index()
            if verbose >= 1:
                display(Markdown("**TOP 10 ANOMALIES**"))
                display(topanomdf.head(10))

            cnt += 1

    # TODO: FIX: This returns the last iteration and should return
    #            the best one.
    if model_type == "lstm_autoencoder":
        # Convert np.array to DF - LSTM-specific
        lossftdf  = pd.DataFrame(lossft, columns=predindf.columns)

        if d4.debug == 5:
            return predinarr, predoutarr, topanomdf, anomsorteddf, losssr, lossftdf
        else:
            return topanomdf, anomsorteddf, losssr, lossftdf
    else:
        return topanomdf, anomsorteddf, losssr

# ML MODELS ###################################################################

def ml_autoencoder(**kwargs): 
    '''
    # GENERAL COMMENTS ========================================================
    #
    # TRAIN / PREDICTION DATA =================================================
    #
    # APPLICABILITY:
    #
    # - This function has been tested at the moment in the following scenarios:
    #   + The prediction dataset includes the train dataset
    #
    # IMPORTANT NOTES:
    #
    # - This function requires 2 dataset inputs: train and prediction
    # - Since this function is an anomaly-oriented implementation test data is 
    #   not valuable, so there will be no test dataset (we will not split the
    #   input training data in the traditional train/test datasets)
    # - The train and prediction datasets can be the same or different.
    #   This is so because you may have the chance to have a clean dataset
    #   without the anomalies you are looking for (e.g. previous to the 
    #   intrusion) or not
    # 
    # NAMING CONVENTIONS: 
    # 
    # For predictions we will use the predindf / predoutdf convention:
    # - predindf  -> Input  data to the model
    # - predoutdf -> Output data from the model (predictions)
    #
    # This in/out differentation is not necessary for the train data, since 
    # there will be no output data from the model after the training process. 
    # So for the train data we will just use: 
    # - traindf   -> Input data to the model
    #
    # =========================================================================
    '''

    # FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO: Analyze carefully if these 2 functions can be merged
    def create_lstm_dataset_train(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])

        if d4.debug >= 3:
            print("")
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
            print("DEBUG: - X  -> "+str(type(X))+" -> "+str(X.shape))
            print("DEBUG: - y  -> "+str(type(y))+" -> "+str(y.shape))
            print("DEBUG: - Xs -> "+str(type(np.array(Xs)))+" -> "+str(np.array(Xs).shape))
            print("DEBUG: - ys -> "+str(type(np.array(ys)))+" -> "+str(np.array(ys).shape))
            print("")

        #return np.array(Xs), np.array(ys)
        return np.array(Xs)

    def create_lstm_dataset_predict(X, time_steps=1):
        Xs = []
        for i in range(len(X) - time_steps + 1):
        #for i in range(0, len(X), time_steps):
            v = X.iloc[i:(i + time_steps)].values
            #print("XXXX: "+str(i)+" - "+str(v))
            Xs.append(v)        
        return np.array(Xs)

    def flatten_3d_array(X):
        '''
        # Input:  X           ->  3D array for lstm, sample x timesteps x features.
        # Output: flattened_X ->  2D array, sample x features.
        '''
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return(flattened_X)

    def prediction_data_preparation(predindf, transform_method, data_scaling_method, verbose):
        print("- Transforming prediction columns ("+transform_method+")")

        if verbose >= 1:
            print("\n[PRED] Before Transform:")
            display(predindf.head(4))

        # This mechanism transforms everything as categoricals
        if transform_method == "categorical_basic":
            # TODO: Implement pandas categorical method. Will probably be the most effective
            # TODO: - https://pbpython.com/categorical-encoding.html
            # TODO: - https://pbpython.com/categorical-encoding.html

            transform_dict = {}
            for col in predindf.columns:
                cats = pd.Categorical(predindf[col]).categories
                d = {}
                for i, cat in enumerate(cats):
                    d[cat] = i
                    transform_dict[col] = d
            
            inverse_transform_dict = {}
            for col, d in transform_dict.items():
                   inverse_transform_dict[col] = {v:k for k, v in d.items()}

            predindf = predindf.replace(transform_dict)

        elif transform_method == "label_encoder":
            transform_dict = {}
            for col in predindf.columns:
                transform_dict[col] = LabelEncoder()
                predindf[col] = transform_dict[col].fit_transform(predindf[col].astype(str))

        else:
            print("ERROR: Invalid transform code: "+transform_method)
            return
        
        if verbose >= 1:
            print("[PRED] After Transform:")
            display(predindf.head(4))
            print("")

        # Scaling - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        scaler = None

        predinarr = np.array(predindf)

        if data_scaling_method != "none":
            if data_scaling_method == "normalize":
                print("- Scaling Data -> Normalize")
                scaler   = MinMaxScaler()
                scaler   = scaler.fit(predinarr)
                predinarr = scaler.transform(predinarr)

                # Summarize the scale of each input variable
                #for i in range(predinarr.shape[1]):
                #    print('>%d, predinarr: min=%.3f, max=%.3f' % (i, predinarr[:, i].min(), predinarr[:, i].max()))

                # TODO: Implement saving
                # Save Scaler Data for future use
                # dump(scaler, open('scaler.pkl', 'wb'))

            elif data_scaling_method == "standardize":
                print("- Scaling Data -> Standardize")
            # TODO: Finish implementation of standardization and normalization
                print("  + WARNING: standardization is not implemented yet. Skipping.")
            #    scaler = {}
            #    for col in traindf.columns:
            #        scaler[col] = StandardScaler()
            #        scaler[col] = scaler.fit(traindf[col])
            #        traindf[col] = label_encoder_train[col].fit_transform(traindf[col])
            #        #train['close'] = scaler.transform(train[['close']])
            #        #test['close']  = scaler.transform(test[['close']])

            if verbose >= 1:
                print("\n[PRED] After Scaling -> "+data_scaling_method)
                #display(pd.DataFrame(predinarr, columns=predindf.columns).head(4))
                display(pd.DataFrame(predinarr, columns=predindf.columns).head(4))

        # LSTM  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if model_type == "lstm_autoencoder":
            print("- [PRED] Creating 3D pred numpy array from predinarr (this can take a long time)...")
            #print("XXXX-A:")
            #print(predinarr)
            predinarr2d = predinarr
            predinarr3d = create_lstm_dataset_predict(pd.DataFrame(predinarr2d), lstm_time_steps)
            predinarr   = predinarr3d
            print("  + "+str(predinarr2d.shape)+" -> "+str(predinarr3d.shape))
            #print("XXXX-B:")
            #print(predinarr)

        return predinarr, transform_dict, scaler

    def train_data_preparation(traindf, transform_method, transform_dict, data_scaling_method, scaler, lstm_time_steps=None, verbose=0):

        # COMMON PREPARATION FOR ALL MODELS - - - - - - - - - - - - - - - - - - - - 
        print("- [TRAIN] Transforming columns ("+transform_method+")")

        if verbose >= 1:
            print("\n[TRAIN] Before Transform:")
            display(traindf.head(4))

        if transform_method == "categorical_basic":
            traindf = traindf.replace(transform_dict)

        elif transform_method == "label_encoder":
            for col in traindf.columns:
                traindf[col] = transform_dict[col].fit_transform(traindf[col].astype(str))
        else:
            print("ERROR: Invalid transform code: "+transform_method)
            return
        
        if verbose >= 1:
            print("\n[TRAIN] After Transform:")
            display(traindf.head(4))

        # Scaling - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
        trainarr = np.array(traindf)

        if d4.debug >= 3:
            print("\n[TRAIN] After Transform (np.array):")
            display(trainarr[:4])

        if data_scaling_method != "none":
            if data_scaling_method == "normalize":
                # Scale the train dataset
                trainarr = scaler.transform(trainarr)

                if d4.debug >= 3:
                    print("")
                    print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
                    print("DEBUG: Per-feature Scaling Min-Max: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                    for i in range(predindf.shape[1]):
                        print('%d, train: min=%.3f, max=%.3f' % (i, trainarr[:, i].min(), trainarr[:, i].max()))
                    print("DEBUG: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    print("")

            elif data_scaling_method == "standardize":
                # TODO: Finish implementation of standardization and normalization
                print("- Scaling Data -> Standardize")
                print("  + WARNING: standardization is not implemented yet. Skipping.")

            if verbose >= 1:
                print("\n[TRAIN] After Scaling -> "+data_scaling_method)
                display(pd.DataFrame(trainarr, columns=traindf.columns).head(4))

        # Model-specific Data Preparation - - - - - - - - - - - - - - - - - - - - - 
        if model_type == "lstm_autoencoder":
            print("- [TRAIN] Creating 3D pred numpy array from predinarr (this can take a long time)...")
            trainarr2d = trainarr
            trainarr3d = create_lstm_dataset_train(pd.DataFrame(trainarr2d), pd.DataFrame(trainarr2d), lstm_time_steps)
            trainarr   = trainarr3d
            print("  + "+str(trainarr2d.shape)+" -> "+str(trainarr3d.shape))

        return trainarr

    # ARGUMENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Generic - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    verbose                = kwargs.get('verbose',                0)
    # Hardware  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    use_gpu                = kwargs.get('use_gpu',                False)
    # Input Data - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    trainorigdf            = kwargs.get('traindf',                None)
    predinorigdf           = kwargs.get('preddf',                 None)
    # Input Data Filtering - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # argtsmin               = kwargs.get('argtsmin',               None)
    # argtsmax               = kwargs.get('argtsmax',               None)
    # Data Preparation - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    transform_method       = kwargs.get('transform_method',       "label_encoder")
    data_scaling_method    = kwargs.get('data_scaling_method',    "normalize")     # { none | normalize | (standardize) }
    # Model Definition - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model_type             = kwargs.get('model_type',             "simple_autoencoder")
    encoding_dim           = kwargs.get('encoding_dim',           3)
    activation_function    = kwargs.get('activation_function',    "relu")
    optimizer              = kwargs.get('optimizer',              None)
    loss                   = kwargs.get('loss',                   None)
    lstm_time_steps        = kwargs.get('lstm_time_steps',        200)
    lstm_units             = kwargs.get('lstm_units',             50)
    # Model Training - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    epochs                 = kwargs.get('epochs',                 40)
    batch_size             = kwargs.get('batch_size',             32)
    loops                  = kwargs.get('loops',                  1)
    # Model Saving - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model_filename         = kwargs.get('model_filename',         None)
    model_filename_root    = kwargs.get('model_filename_root',    None)
    autosave_minloss_model = kwargs.get('autosave_minloss_model', False)
    # Predictions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    error_threshold        = kwargs.get('error_threshold',          None)
    error_ntop             = kwargs.get('error_ntop',               None)
    
    # TODO: Implement filtering by time
    # ToDO: Filtering by time is not supported at this time. Maybe at a later stage...
    # ToDo: dftsmin = df.index.min()
    # ToDo: dftsmax = df.index.max()

    # ToDo: if argtsmin is None:
    # ToDo:     tsmin=dftsmin
    # ToDo: else:
    # ToDo:     tsmin=argtsmin

    # ToDo: if argtsmax is None:
    # ToDo:     tsmax=dftsmax
    # ToDo: else:
    # ToDo:     tsmax=argtsmax

    # BODY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # HEALTH CHECKS -----------------------------------------------------------
    if model_type is None:
        print("ERROR: model_type not selected.")
        print("Options: { simple_autoencoder | multilayer_autoencoder }")
        return

    # CREATE A COPY OF ORIGINAL DATA FOR MANIPULATION -------------------------
    #
    # Both traindf and predindf will be modified through the Data Preparation
    # phases (encoding, normalizing, etc.), this is because we save the original
    # data as:
    # - trainorigdf  -> Input train data       (as provided to the function)
    # - predinorigdf -> Input predictions data (as provided to the function)

    traindf   = trainorigdf.copy()
    predindf  = predinorigdf.copy()

    # ARGUMENT PROCESSING -----------------------------------------------------
    if traindf is not None:
        traindf_shape = traindf.shape
    else:
        traindf_shape = None

    if predindf is not None:
        predindf_shape = predindf.shape
    else:
        predindf_shape = None

    # INFO --------------------------------------------------------------------
    print("- General:")
    print("  + Verbosity:               "+str(verbose))
    print("- Input Data:")       
    print("  + Train DF:                "+str(traindf_shape))
    print("  + Prediction DF:           "+str(predindf_shape))
    print("- Data Preparation:")
    print("  + Transform Method:        "+transform_method)
    print("  + Data Scaling Method:     "+str(data_scaling_method))
    print("- Model Parameters:")
    print("  + Model Type:              "+model_type)
    print("  + Encoding Dimension:      "+str(encoding_dim))
    print("  + Activation Function:     "+str(activation_function))
    # Model-specific Arguments
    if model_type == "lstm_autoencoder":
        print("  + lstm_units:              "+str(lstm_units))
        print("  + lstm_time_steps:         "+str(lstm_time_steps))
    print("- Training Parameters:") 
    print("  + Training Loops:          "+str(loops))
    print("  + epochs:                  "+str(epochs))
    print("  + batch_size:              "+str(batch_size))
    print("- Model Saving:")         
    print("  + autosave_minloss_model:  "+str(autosave_minloss_model))
    print("  + model_filename:          "+str(model_filename))
    print("  + model_filename_root:     "+str(model_filename_root))
    print("- Predictions:")        
    print("  + error_threshold:         "+str(error_threshold))
    print("  + error_ntop:              "+str(error_ntop))
    print("")

    # INITIALIZATION ----------------------------------------------------------
    np.random.seed(8)

    runid = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    # DATA PREPARATION --------------------------------------------------------
    # We will do Data Preparation for the prediction data first, in order to 
    # make sure there are no outliers which may break the process later
    # In the current implementation of this function, the train dataset is
    # included in the prediction dataset, so the right dataset to define the 
    # transform matrixes (transform, standardization, normalization, etc.) is 
    # the preddf. And then, we will apply that to the traindf (which will be a
    # subset of the preddf as we say)
  
    # DATA SUBSETS  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if model_type == "lstm_autoencoder":
        predinoriglstmdf = predinorigdf.iloc[lstm_time_steps-1:]

    # PREDICTION  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    predinarr, transform_dict, scaler = prediction_data_preparation(predindf, transform_method, data_scaling_method, verbose)
    #print("XXXX: "+str(predinarr.shape))


    # CREATING THE NEURAL NETWORK ARCHITECTURE --------------------------------
    # Load model, if model file exists
    if model_filename is not None and os.path.exists(model_filename):
            model_loaded_from_file = True

            print("- Loading Model from:")
            print("    "+model_filename)
            autoencoder = load_model(model_filename)

            print("")
            print("Autoencoder Summary:")
            print("")
            autoencoder.summary()
            print("")

    else:
        # TRAIN - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        trainarr = train_data_preparation(traindf, transform_method, transform_dict, data_scaling_method, scaler, lstm_time_steps, verbose)

        model_loaded_from_file = False

        # DATA PREPARATION --------------------------------------------------------

        # SELECTING TRAINING / TEST DATA ------------------------------------------
        # Not useful in unsupervised anomaly detection. Skipping.

        # ML MODEL CREATION -------------------------------------------------------
        nfeatures    = traindf.shape[1]

        models   = {}
        mdlnames = []
        losses   = []
        tlcnt    = 1
        while tlcnt <= loops:
            print("")
            print("[LOOP ITERATION: "+str(tlcnt)+"/"+str(loops)+"]")
            print("")

            print("MODEL CREATION")
            print("")


            # We create an encoder and decoder. 
            # The ReLU function, which is a non-linear activation function, is used in the encoder. 
            # The encoded layer is passed on to the decoder, where it tries to reconstruct the input data pattern

            if model_type == "simple_autoencoder":
                input_dim    = nfeatures
                hidden_dim   = encoding_dim

                print("- Creating Model")
                print("  + No. Features:          "+str(nfeatures))
                print("  + Input Array Dimension: "+str(input_dim))
                print("")

                input_layer = Input(shape=(input_dim,))
                print(input_layer)
                encoded     = Dense(encoding_dim, activation='relu'  )(input_layer)
                decoded     = Dense(input_dim,    activation='linear')(encoded)

                autoencoder = Model(input_layer, decoded)

                optimizer_deflt = 'adadelta'
                loss_deflt      = 'mse'

            elif model_type == "multilayer_autoencoder":
                input_dim    = nfeatures
                hidden_dim   = encoding_dim

                print("- Creating Model")
                print("  + No. Features:    "+str(nfeatures))
                print("  + Input Array Dimension: "+str(input_dim))
                print("")

                input_layer = Input(shape=(input_dim,))
                encoded     = Dense(encoding_dim, activation="relu"  )(input_layer)
                encoded     = Dense(hidden_dim,   activation="relu"  )(encoded)
                decoded     = Dense(hidden_dim,   activation="relu"  )(encoded)
                decoded     = Dense(encoding_dim, activation="relu"  )(decoded)
                decoded     = Dense(input_dim,    activation="linear")(decoded)

                autoencoder = Model(input_layer, decoded)

                optimizer_deflt = 'adadelta'
                loss_deflt      = 'mse'

            elif model_type == "lstm_autoencoder":
                # Refs: - https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
                #       - https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/

                print("- Creating Model")
                print("  + No. Features:          "+str(nfeatures))
                print("  + Input Array Dimension: "+str(trainarr.shape))
                print("")

                # SINGLE LAYER - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
                #
                # For LSTM use with GPUs, see:
                # - https://keras.io/api/layers/recurrent_layers/lstm/
                # - https://stackoverflow.com/questions/62044838/using-cudnn-kernel-for-lstm

                autoencoder = Sequential()

                if use_gpu :
                    autoencoder.add(LSTM(lstm_units, activation=activation_function, input_shape=(lstm_time_steps, nfeatures),
                                         recurrent_activation="sigmoid", recurrent_dropout=0.0, unroll=False, use_bias=True))
                else:
                    autoencoder.add(LSTM(lstm_units, activation=activation_function, input_shape=(lstm_time_steps, nfeatures)))
                    
                autoencoder.add(RepeatVector(n=lstm_time_steps))
                autoencoder.add(LSTM(lstm_units, return_sequences=True))
                autoencoder.add(TimeDistributed(Dense(nfeatures)))

                # MULTILAYER - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #  autoencoder = keras.Sequential()
                #  # Encoder
                #  autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True))
                #  autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
                #  autoencoder.add(RepeatVector(timesteps))
                #  # Decoder
                #  autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
                #  autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
                #  autoencoder.add(TimeDistributed(Dense(input_dim)))

                # With Dropout Layers  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #  model = keras.Sequential()
                #  model.add(keras.layers.LSTM( units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
                #  model.add(keras.layers.Dropout(rate=0.2))
                #  model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
                #  model.add(keras.layers.LSTM(units=64, return_sequences=True))
                #  model.add(keras.layers.Dropout(rate=0.2))
                #  model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
                #  model.compile(loss='mae', optimizer='adam')

                optimizer_deflt = 'adam'
                loss_deflt      = 'mae'
           
            if optimizer is None:
                optimizer = optimizer_deflt

            if loss is None:
                loss = loss_deflt

            # The following model maps the input to its reconstruction, which is done in the decoder layer, decoded.
            # Next, the optimizer and loss function is defined using the compile method.
            # The adadelta optimizer uses exponentially-decaying gradient averages and is a highly-adaptive learning rate method.
            # The reconstruction is a linear process and is defined in the decoder using the linear activation function.
            # The loss is defined as mse, which is mean squared error

            print("- Compiling Model")
            print("")
            autoencoder.compile(optimizer=optimizer, loss=loss)
            modelname = autoencoder.name

            # Save autoencoder name to list
            mdlnames.append(autoencoder.name)
            # Save autoencoder model to dict
            models[modelname] = autoencoder

            display(Markdown("**Autoencoder Summary:**"))
            print("")
            autoencoder.summary()
            print("")

            # TRAINING THE NETWORK --------------------------------
            print("TRAINING")
            print("")
            print("- Training Info:")
            print("  + epochs     = "+str(epochs))
            print("  + batch_size = "+str(batch_size))
            print("")

            
            now = datetime.datetime.now()
            print("- Training Start: "+str(now.strftime("%Y-%m-%d %H:%M:%S")))
            print("")

            losshist = autoencoder.fit(trainarr, trainarr, epochs=epochs, batch_size=batch_size)
            losses.append(losshist.history['loss'][-1])
            tlcnt += 1

            now = datetime.datetime.now()
            print("")
            print("- Training End:   "+str(now.strftime("%Y-%m-%d %H:%M:%S")))
            print("")

        minloss        = min(losses)
        minlossidx     = losses.index(minloss)
        minlossmdlname = mdlnames[minlossidx]
        minlossmodel   = models[minlossmdlname]

        print("")
        print("- Models:          "+str(mdlnames))
        print("- Losses:          "+str(losses))
        print("- Min. Loss:       "+str(minloss))
        print("- Min. Loss Model: "+str(minlossmdlname))
        print("")
    
        # TODO: Remove - NOT NEEDED
        #    # Undo the encoding (if needed)
        #    if transform_method == "label_encoder":
        #        for col in traindf.columns:
        #            traindf[col] = label_encoder_train[col].inverse_transform(traindf[col])

        # Save model, if model file does not exist
        if autosave_minloss_model:
            if autosave_minloss_model:
                minlossautoencoder = minlossmodel
                if model_filename_root is not None:
                    model_filename_root = model_filename_root
                else:
                    model_filename_root = ""

                model_filename_save = model_filename_root+"-"+model_type+"-"+runid+"-loss_"+str(minloss)+".h5"

                if not os.path.exists(model_filename_save):
                    # TODO: If filename does not have an .h5 extension, add it
                    print("- Saving Model to:")
                    print("    "+model_filename_save)
                    print("")
                    autoencoder.save(model_filename_save)
            else:
                if not os.path.exists(model_filename):
                    # TODO: If filename does not have an .h5 extension, add it
                    print("- Saving Model to:")
                    print("    "+model_filename)
                    autoencoder.save(model_filename)
                    print("")

    # DO PREDICTIONS / FIND ANOMALIES =========================================

    # Once the model is fitted, we predict the input values by passing the same X_train dataset to the autoencoder's predict method.
    # Next, we calculate the mse values to know whether the autoencoder was able to reconstruct the dataset correctly and how much the reconstruction error was:

    print("PREDICTIONS")
    print("")

    # RUN MODEL ---------------------------------------------------------------
    if not model_loaded_from_file:
        # Set autoencoder model to the one with minimal loss
        print("- Setting autoencoder to min loss model: "+minlossmdlname)
        autoencoder = minlossmodel
        
    print("- Input DF Shape: "+str(predindf.shape)+" -> "+str(predinarr.shape))

    print("- Running predictions...")
    now = datetime.datetime.now()
    print("  + Start: "+str(now.strftime("%Y-%m-%d %H:%M:%S")))
    predoutarr = autoencoder.predict(predinarr)
    now = datetime.datetime.now()
    print("  + End:   "+str(now.strftime("%Y-%m-%d %H:%M:%S")))

    # Calculate Loss  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if   model_type == "simple_autoencoder":
        loss = np.mean(np.power(predinarr - predoutarr, 2), axis=1)
        predinfindf = predindf
    elif model_type == "multilayer_autoencoder":
        loss = np.mean(np.power(predinarr - predoutarr, 2), axis=1)
        predinfindf = predindf
    elif model_type == "lstm_autoencoder":

        predinarr_flat  = flatten_3d_array(predinarr)
        predoutarr_flat = flatten_3d_array(predoutarr)

        #print("XXXX-C:")
        #print(predinarr)
        #print(pd.DataFrame(predinarr_flat).value_counts())

        #if d4.debug >= 5:
        #   display(predinorigdf.value_counts())
        #   display(pd.DataFrame(predinarr_flat).value_counts())
        #   display(pd.DataFrame(predoutarr_flat).value_counts())

        ## Recover predictions in categorical way 
        ## 1. Undo the normalizing
        #if data_scaling_method != "none":
        #    if data_scaling_method == "normalize":
        #        # TODO: Reverse this
        #        predout_decodedarr = scaler.inverse_transform(predoutarr_flat)
        #    elif data_scaling_method == "standardize":
        #        print("- Un-Scaling Data -> Standardize")
        #        print("  + WARNING: standardization is not implemented yet. Skipping.")
        #    
        #    predout_decodeddf = pd.DataFrame(predout_decodedarr, columns=predindf.columns)
        #
        ## 2. Undo the encoding 
        #if transform_method == "label_encoder":
        #    for col in predindf.columns:
        #        predout_decodeddf[col] = transform_dict[col].inverse_transform(predout_decodeddf[col])
        #        print("XXXX: "+str(type(predout_decodeddf)))

        loss_ft   = np.power(predinarr_flat - predoutarr_flat, 2)
        loss_full = np.mean(np.power(predinarr_flat - predoutarr_flat, 2), axis=1)

        loss      = loss_full

        predinfindf = predindf.iloc[lstm_time_steps:]

        if d4.debug >= 3:
            print("")
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
            print("DEBUG: - predindf:        "+str(predindf.shape))
            print("DEBUG: - predinfindf:     "+str(predinfindf.shape))
            print("DEBUG: - predinarr:       "+str(predinarr.shape))
            print("DEBUG: - predinarr_flat:  "+str(predinarr_flat.shape))
            print("DEBUG: - predoutarr:      "+str(predoutarr.shape))
            print("DEBUG: - predoutarr_flat: "+str(predoutarr_flat.shape))
            print("DEBUG: - loss :           "+str(loss.shape))
            print("")

    plt.plot(loss)

    # OBTAIN ANOMALIES --------------------------------------------------------

#test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
#test_score_df['loss'] = test_mae_loss

    # Auto-calculate error_threshold - Top N values
    if error_threshold is None:
        if error_ntop is None:
            ntoperror = 15
        else:
            # Ensure that the number of error rows specified are <= DF rows
            if error_ntop >= predinfindf.shape[0]:
                ntoperror = predinfindf.shape[0]
            else:
                ntoperror = error_ntop
        print(ntoperror)
        error_threshold = int(np.sort(loss)[-ntoperror])
        autocalcmsg = " (Auto-calculated - Top "+str(ntoperror)+")"
    else:
        autocalcmsg = ""

    print("- Error Threshold: "+str(error_threshold)+autocalcmsg)


    # Select entries above the error threshold
    # Instead of using predoutdf we will use predinfindf, which has been adapted
    # to be the same dimension as predoutdf and is good enough to identify the
    # anomalous entries as predoutdf, since they both have corresponding entries
    # (original vs predicted) in a 1-to-1 way
    if   model_type == "simple_autoencoder":
        anomdf = predinorigdf[loss >= error_threshold]
    elif model_type == "multilayer_autoencoder":
        anomdf = predinorigdf[loss >= error_threshold]
    elif model_type == "lstm_autoencoder":
        anomdf = predinoriglstmdf[loss >= error_threshold]

    if d4.debug >= 3:
        print("")
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
        print("DEBUG: anomdf shape: "+str(anomdf.shape))
        print("DEBUG: anomdf dtype & head: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        display(anomdf.dtypes)
        display(anomdf.head(4))
        print("DEBUG: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("")

    print("- No.Anomalies: "+str(len(anomdf)))
    print("- RUN ID:       "+str(runid))

    # TODO: FIX: For some reason this is broken.
    #display(pd.DataFrame(anom.groupby(df.columns).size()).rename(columns={0: 'Count'}))

    # RETURN ------------------------------------------------------------------
    losssr = pd.Series(loss)

    if model_type == "lstm_autoencoder":
        if d4.debug == 5:
            return predinarr_flat, predoutarr_flat, anomdf, losssr, loss_ft
        else:
            return anomdf, losssr, loss_ft
    else:
        return anomdf, losssr

    # TODO: DELETE: OLD STUFF vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    #   # Auto-calculate error_threshold - Top N values
    #   if mse_threshold is None:
    #       if mse_ntop is None:
    #           ntopmse = 15
    #       else:
    #           if mse_ntop >= preddf.shape[0]:
    #               ntopmse = preddf.shape[0]
    #           else:
    #               ntopmse = mse_ntop

    #       mse_threshold = int(np.sort(mse)[-ntopmse])
    #       autocalcmsg = " (Auto-calculated - Top "+str(ntopmse)+")"
    #   else:
    #       autocalcmsg = ""

    #   print("- MSE Threshold: "+str(mse_threshold)+autocalcmsg)

    #   # Reverse Transformation
    #   if   transform_method == "categorical_basic":
    #       preddf = preddf.replace(inverse_transform_dict)
    #   elif transform_method == "label_encoder":
    #       for col in preddf.columns:
    #           preddf[col] = label_encoder_pred[col].inverse_transform(preddf[col])

    #   # Select entries above the mse threshold
    #   anomdf = preddf[mse >= mse_threshold]

    #   print("- No.Anomalies: "+str(len(anomdf)))

    #   # TODO: FIX: For some reason this is broken.
    #   #display(pd.DataFrame(anom.groupby(df.columns).size()).rename(columns={0: 'Count'}))

    #   return anomdf, msesr

## ML - AutoEncoder ============================================================
#def ml_autoencoder_old(df, preddf, **kwargs): 
#
#    argtsmin        = kwargs.get('argtsmin',        None)
#    argtsmax        = kwargs.get('argtsmax',        None)
#    mse_threshold   = kwargs.get('mse_threshold',   None)
#    mse_ntop        = kwargs.get('mse_ntop',        None)
#    epochs          = kwargs.get('epochs',          40)
#    model_type      = kwargs.get('model_type',      None)
#    model_filename  = kwargs.get('model_filename',  None)
#   
#    # Filtering by time is not supported at this time. Maybe at a later stage...
#    #dftsmin = df.index.min()
#    #dftsmax = df.index.max()
#
#    #if argtsmin is None:
#    #    tsmin=dftsmin
#    #else:
#    #    tsmin=argtsmin
#
#    #if argtsmax is None:
#    #    tsmax=dftsmax
#    #else:
#    #    tsmax=argtsmax
#
#    if model_type is None:
#        print("ERROR: model_type not selected.")
#        print("Options: { simple_autoencoder | multilayer_autoencoder }")
#        return
#
#    print("- Model Type: "+model_type)
#
#    np.random.seed(8)
#
#    # DATA PREPARATION ------------------------------------
#    print("- Transforming columns")
#    transform_dict = {}
#    for col in df.columns:
#        cats = pd.Categorical(df[col]).categories
#        d = {}
#        for i, cat in enumerate(cats):
#            d[cat] = i
#            transform_dict[col] = d
#    
#    inverse_transform_dict = {}
#    for col, d in transform_dict.items():
#           inverse_transform_dict[col] = {v:k for k, v in d.items()}
#    
#    df = df.replace(transform_dict)
#
#    # SELECTING TRAINING / TEST DATA ----------------------
#    print("- Splitting Train / Test")
#    X = df
#    from sklearn.model_selection import train_test_split
#    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
#    
#    print("- Splitting Input Data:")
#    print("  + X         -> "+str(X.shape))
#    print("    - X_train -> "+str(X_train.shape))
#    print("    - X_test  -> "+str(X_test.shape))
#
#    # SELECTING TRAINING / TEST DATA &         ------------
#    # CREATING THE NEURAL NETWORK ARCHITECTURE ------------
#    # There are 4 different input features, and as we plan to use all the features in the autoencoder,
#    # we define the number of input neurons to be 4.
#    print("- Creating Model")
#
#    # Load model, if model file exists
#    if model_filename is not None and os.path.exists(model_filename):
#            print("- Loading Model from:")
#            print("    "+model_filename)
#            autoencoder = load_model(model_filename)
#
#            print("")
#            display(("Autoencoder Summary:"))
#            print("")
#            autoencoder.summary()
#            print("")
#    else:
#
#        nfeatures    = df.shape[1]
#
#        input_dim    = X_train.shape[1]
#        #encoding_dim = nfeatures - 2
#        encoding_dim = 3
#        hidden_dim   = encoding_dim
#
#        print("- No. Features:    "+str(nfeatures))
#        print("- Input Dimension: "+str(input_dim))
#
#        # Input Layer
#        input_layer  = Input(shape=(input_dim,))
#
#        # We create an encoder and decoder. 
#        # The ReLU function, which is a non-linear activation function, is used in the encoder. 
#        # The encoded layer is passed on to the decoder, where it tries to reconstruct the input data pattern
#
#        if model_type == "simple_autoencoder":
#            encoded = Dense(encoding_dim, activation='relu')(input_layer)
#            decoded = Dense(nfeatures, activation='linear')(encoded)
#        elif model_type == "multilayer_autoencoder":
#            encoded = Dense(encoding_dim, activation="relu")(input_layer)
#            encoded = Dense(hidden_dim,   activation="relu")(encoded)
#            decoded = Dense(hidden_dim,   activation="relu")(encoded)
#            decoded = Dense(encoding_dim, activation="relu")(decoded)
#            decoded = Dense(input_dim,    activation="linear")(decoded)
#
#        # The following model maps the input to its reconstruction, which is done in the decoder layer, decoded.
#        # Next, the optimizer and loss function is defined using the compile method.
#        # The adadelta optimizer uses exponentially-decaying gradient averages and is a highly-adaptive learning rate method.
#        # The reconstruction is a linear process and is defined in the decoder using the linear activation function.
#        # The loss is defined as mse, which is mean squared error
#
#        autoencoder = Model(input_layer, decoded)
#        autoencoder.compile(optimizer='adadelta', loss='mse')
#
#        print("")
#        display(Markdown("**Autoencoder Summary:**"))
#        print("")
#        autoencoder.summary()
#        print("")
#
#        ## LSTM vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#        #lstm_autoencoder = Sequential()
#        ## Encoder
#        #lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
#        #lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
#        #lstm_autoencoder.add(RepeatVector(timesteps))
#        ## Decoder
#        #lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
#        #lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
#        #lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
#        #
#        #lstm_autoencoder.summary()
#        ##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#        # TRAINING THE NETWORK --------------------------------
#        # The training data, X_train, is fitted into the autoencoder. 
#        # Let's train our autoencoder for 100 epochs with a batch_size of 4 and observe if it reaches a stable train or test loss value
#        
#        batch_size=4
#
#        print("- Training:")
#        print("  + epochs     = "+str(epochs))
#        print("  + batch_size = "+str(batch_size))
#
#        #X_train = np.array(X_train)
#        # TODO: FIX: Manual Testing JG
#        X_train = np.array(X)
#        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)
#
#        # Save model, if model file does not exist
#        if model_filename is not None:
#            if not os.path.exists(model_filename):
#                # TODO: If filename does not have an .h5 extension, add it
#                print("- Saving Model to:")
#                print("    "+model_filename)
#                autoencoder.save(model_filename)
#
#    # DOING PREDICTIONS -----------------------------------
#
#    # Once the model is fitted, we predict the input values by passing the same X_train dataset to the autoencoder's predict method.
#    # Next, we calculate the mse values to know whether the autoencoder was able to reconstruct the dataset correctly and how much the reconstruction error was:
#
#    # PREDICTION DATA PREPARATION -------------------------
#    print("- Transforming prediction columns")
#    transform_dict = {}
#    for col in preddf.columns:
#        cats = pd.Categorical(preddf[col]).categories
#        d = {}
#        for i, cat in enumerate(cats):
#            d[cat] = i
#            transform_dict[col] = d
#    
#    inverse_transform_dict = {}
#    for col, d in transform_dict.items():
#           inverse_transform_dict[col] = {v:k for k, v in d.items()}
#    
#    preddf = preddf.replace(transform_dict)
#
#    print("- Doing Predictions...\n")
#
#    print("- Predictions:")
#
#    # TODO: FIX: We should probably predict over a different dataset
#    #            or the full dataset, not only on the train dataset
#    #predictions = autoencoder.predict(X)
#    #mse = np.mean(np.power(X - predictions, 2), axis=1)
#    predictions = autoencoder.predict(preddf)
#    mse = np.mean(np.power(preddf - predictions, 2), axis=1)
#
#    plt.plot(mse)
#
#    # Auto-calculate mse_threshold - Top N values
#    if mse_threshold is None:
#        if mse_ntop is None:
#            ntopmse = 15
#        else:
#            ntopmse = mse_ntop
#
#        mse_threshold = int(np.sort(mse)[-ntopmse])
#        autocalcmsg = " (Auto-calculated - Top "+str(ntopmse)+")"
#    else:
#        autocalcmsg = ""
#
#    print("- MSE Threshold: "+str(mse_threshold)+autocalcmsg)
#
#    # Select entries above the mse threshold
#    xxx = preddf[mse >= mse_threshold]
#
#    xxxdf = pd.DataFrame(xxx)
#    xxxdf.columns = df.columns
#    anom = xxxdf.replace(inverse_transform_dict)
#    print("- No.Anomalies: "+str(len(xxxdf)))
#
#    # TODO: FIX: For some reason this is broken.
#    #display(pd.DataFrame(anom.groupby(df.columns).size()).rename(columns={0: 'Count'}))
#
#    return anom, mse
#
## ML - Access Anomalies =======================================================
#def ml_access_anomalies(secevtxdf, **kwargs): 
#
#    argtsmin        = kwargs.get('argtsmin',        None)
#    argtsmax        = kwargs.get('argtsmax',        None)
#    mse_threshold   = kwargs.get('mse_threshold',   None)
#    mse_ntop        = kwargs.get('mse_ntop',        None)
#    epochs          = kwargs.get('epochs',          40)
#    model_type      = kwargs.get('model_type',      None)
#    model_filename  = kwargs.get('model_filename',  None)
#   
#    dftsmin=secevtxdf.index.min()
#    dftsmax=secevtxdf.index.max()
#
#    if argtsmin is None:
#        tsmin=dftsmin
#    else:
#        tsmin=argtsmin
#
#    if argtsmax is None:
#        tsmax=dftsmax
#    else:
#        tsmax=argtsmax
#
#    evts4624 = secevtxdf.query('EventID_ == 4624', engine="python")
#
#    np.random.seed(8)
#
#    # DATA PREPARATION ------------------------------------
#    evts4624_nonsysusers = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')].reset_index()
#    useraccess           = evts4624_nonsysusers[["Timestamp", "Computer", "WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('Timestamp')
#    this_useraccess      = useraccess.loc[tsmin:tsmax]
#    
#    user_access_uwil = this_useraccess[['Computer', 'TargetUserName',"WorkstationName","IpAddress",'LogonType']].copy()
#    
#    # Lower-case cols
#    user_access_uwil['Computer']        = user_access_uwil['Computer'].str.lower()
#    user_access_uwil['WorkstationName'] = user_access_uwil['WorkstationName'].str.lower().fillna("null_workstation")
#    user_access_uwil['TargetUserName']  = user_access_uwil['TargetUserName'].str.lower()
#    user_access_uwil['LogonType']       = user_access_uwil['LogonType'].astype(str)
#    
#    user_access_uwil_str = user_access_uwil.copy()
#    
#    user_access_uwil_str['TU-WN-IP-LT'] = "[" + user_access_uwil['Computer'] + "]" + "[" + user_access_uwil['TargetUserName'] + "]" + "[" + user_access_uwil['IpAddress'] + "][" + user_access_uwil['LogonType'] + "]"
#    user_access_uwil_str.drop(columns=['Computer', 'WorkstationName', 'IpAddress', 'TargetUserName', 'LogonType'], inplace=True)
#    user_access_uwil_str = user_access_uwil_str.sort_values(by='TU-WN-IP-LT')
#
#    df = user_access_uwil
#
#    transform_dict = {}
#    for col in df.columns:
#        cats = pd.Categorical(df[col]).categories
#        d = {}
#        for i, cat in enumerate(cats):
#            d[cat] = i
#            transform_dict[col] = d
#    
#    inverse_transform_dict = {}
#    for col, d in transform_dict.items():
#           inverse_transform_dict[col] = {v:k for k, v in d.items()}
#    
#    df = df.replace(transform_dict)
#
#    # SELECTING TRAINING / TEST DATA ----------------------
#    X = df
#    from sklearn.model_selection import train_test_split
#    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
#    
#    print("- Splitting Input Data:")
#    print("  + X         -> "+str(X.shape))
#    print("    - X_train -> "+str(X_train.shape))
#    print("    - X_test  -> "+str(X_test.shape))
#
#    # SELECTING TRAINING / TEST DATA &         ------------
#    # CREATING THE NEURAL NETWORK ARCHITECTURE ------------
#    # There are 4 different input features, and as we plan to use all the features in the autoencoder,
#    # we define the number of input neurons to be 4.
#
#    # Load model, if model file exists
#    if model_filename is not None and os.path.exists(model_filename):
#            print("- Loading Model from:")
#            print("    "+model_filename)
#            autoencoder = load_model(model_filename)
#
#            print("")
#            print("Autoencoder Summary:**")
#            print("")
#            autoencoder.summary()
#            print("")
#    else:
#
#        nfeatures    = 5
#
#        input_dim    = X_train.shape[1]
#        encoding_dim = nfeatures - 2
#        hidden_dim   = encoding_dim
#
#        print("- No. Features:    "+str(nfeatures))
#        print("- Input Dimension: "+str(input_dim))
#
#        # Input Layer
#        input_layer  = Input(shape=(input_dim,))
#
#        # We create an encoder and decoder. 
#        # The ReLU function, which is a non-linear activation function, is used in the encoder. 
#        # The encoded layer is passed on to the decoder, where it tries to reconstruct the input data pattern
#
#        if model_type == "simple_autoencoder":
#            encoded = Dense(encoding_dim, activation='relu')(input_layer)
#            decoded = Dense(nfeatures, activation='linear')(encoded)
#        elif model_type == "multi_layer_autoencoder":
#            encoded = Dense(encoding_dim, activation="relu")(input_layer)
#            encoded = Dense(hidden_dim,   activation="relu")(encoded)
#            decoded = Dense(hidden_dim,   activation="relu")(encoded)
#            decoded = Dense(encoding_dim, activation="relu")(decoded)
#            decoded = Dense(input_dim,    activation="linear")(decoded)
#
#        # The following model maps the input to its reconstruction, which is done in the decoder layer, decoded.
#        # Next, the optimizer and loss function is defined using the compile method.
#        # The adadelta optimizer uses exponentially-decaying gradient averages and is a highly-adaptive learning rate method.
#        # The reconstruction is a linear process and is defined in the decoder using the linear activation function.
#        # The loss is defined as mse, which is mean squared error
#
#        autoencoder = Model(input_layer, decoded)
#        autoencoder.compile(optimizer='adadelta', loss='mse')
#
#        print("")
#        display(Markdown("**Autoencoder Summary:**"))
#        print("")
#        autoencoder.summary()
#        print("")
#
#        ## LSTM vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#        #lstm_autoencoder = Sequential()
#        ## Encoder
#        #lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
#        #lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
#        #lstm_autoencoder.add(RepeatVector(timesteps))
#        ## Decoder
#        #lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
#        #lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
#        #lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
#        #
#        #lstm_autoencoder.summary()
#        ##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#        # TRAINING THE NETWORK --------------------------------
#        # The training data, X_train, is fitted into the autoencoder. 
#        # Let's train our autoencoder for 100 epochs with a batch_size of 4 and observe if it reaches a stable train or test loss value
#        
#        batch_size=4
#
#        print("- Training:")
#        print("  + epochs     = "+str(epochs))
#        print("  + batch_size = "+str(batch_size))
#
#        X_train = np.array(X_train)
#        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)
#
#        # Save model, if model file does not exist
#        if model_filename is not None:
#            if not os.path.exists(model_filename):
#                # TODO: If filename does not have an .h5 extension, add it
#                print("- Saving Model to:")
#                print("    "+model_filename)
#                autoencoder.save(model_filename)
#
#    # DOING PREDICTIONS -----------------------------------
#
#    # Once the model is fitted, we predict the input values by passing the same X_train dataset to the autoencoder's predict method.
#    # Next, we calculate the mse values to know whether the autoencoder was able to reconstruct the dataset correctly and how much the reconstruction error was:
#
#    print("- Predictions:")
#
#    # TODO: FIX: We should probably predict over a different dataset
#    #            or the full dataset, not only on the train dataset
#    #predictions = autoencoder.predict(X)
#    #mse = np.mean(np.power(X - predictions, 2), axis=1)
#    predictions = autoencoder.predict(X_train)
#    mse = np.mean(np.power(X_train - predictions, 2), axis=1)
#
#
#    plt.plot(mse)
#
#    # Auto-calculate mse_threshold - Top N values
#    if mse_threshold is None:
#        if mse_ntop is None:
#            ntopmse = 15
#        else:
#            ntopmse = mse_ntop
#
#        mse_threshold = int(np.sort(mse)[-ntopmse])
#        autocalcmsg = " (Auto-calculated - Top "+str(ntopmse)+")"
#    else:
#        autocalcmsg = ""
#
#    print("- MSE Threshold: "+str(mse_threshold)+autocalcmsg)
#
#    # Select entries above the mse threshold
#    xxx = X_train[mse >= mse_threshold]
#
#    xxxdf = pd.DataFrame(xxx)
#    xxxdf.columns = ['Computer', 'TargetUserName', 'WorkstationName', 'IpAddress', 'LogonType']
#    anom = xxxdf.replace(inverse_transform_dict)
#    print("- No.Anomalies: "+str(len(xxxdf)))
#    display(pd.DataFrame(anom.groupby(['Computer', 'WorkstationName', 'IpAddress', 'TargetUserName', 'LogonType']).size()).rename(columns={0: 'Count'}))
#
#    anom_uwil = anom.copy()
#
#    anom_uwil['TU-WN-IP-LT'] = "["+anom_uwil['Computer']+"]["+anom_uwil['TargetUserName']+"]["+anom_uwil['IpAddress']+"]["+anom_uwil['LogonType']+"]"
#    #anom_uwil.drop(columns=['WorkstationName','IpAddress','TargetUserName','LogonType'],inplace=True)
#
#    anom_uwil_uniq_df = pd.DataFrame(anom_uwil['TU-WN-IP-LT'].unique(),columns=['TU-WN-IP-LT']).sort_values(by='TU-WN-IP-LT')
#    display(anom_uwil_uniq_df)
#
#    anom_uwil_uniq_df_ts = user_access_uwil_str[user_access_uwil_str['TU-WN-IP-LT'].isin(anom_uwil_uniq_df['TU-WN-IP-LT'])]
#    #anom_uwil_uniq_df_ts.head(2)
#
#    # OVERPLOT ANOMALOUS DATA OVER ORIGINAL DATA ------------------------------
#    col='TU-WN-IP-LT'
#    label=col
#    data=user_access_uwil_str
#
#    fig = plt.figure()
#    plt.figure(figsize=(20,10))
#
#    # Plot original data (green)
#    frame = data
#    plt.grid(color='g', linestyle='-', linewidth=0.1)
#    plt.plot(frame.index, data[col], 'g.')
#
#    # Over-Plot anomalous data (red)
#    frame = anom_uwil_uniq_df_ts
#    plt.plot(frame.index, frame[col], 'r.')
#
#    plt.show()
#
#    return X_train, mse
#
