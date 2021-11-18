
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4mctm

###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import os
import glob
import re
import time
import inspect
import xmltodict
import json
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as et

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.gui    as d4gui
import ds4n6_lib.utils  as d4utl

###############################################################################
# IDEAS
###############################################################################
# is_deleted()
# is_file()
# is_dir() / is_folder() - level
# ext() # filter by Extension
# nofn  # exclude $FILE_NAME entries

###############################################################################
# FUNCTIONS
###############################################################################

# FILE READING FUNCTIONS ######################################################

# FILE READING FUNCTIONS ######################################################

def read_data(evdl, **kwargs):
    if d4.debug >= 3:
        print("DEBUG: [mctm] read_data")

    return d4com.read_data_common(evdl, **kwargs)

#def read_fstl(fstlf, windows=False, dateindex=True, adjust=True, harmonize=True):
#    print("- Reading "+fstlf)
#    fstldf = pd.read_csv(fstlf)
#
#    fstltype = fstldf.d4.df_source_identify()
#
#    # Adjust ------------------------------------------------------------------
#    if adjust :
#        if fstltype == "pandas_dataframe-fstl-mactime-raw":
#            print("- Adjusting data types")
#            fstldf['Date'] = fstldf['Date'].astype('datetime64')
#
#            fstldf = fstldf.rename(columns={"File Name": "FileName"})
#            fstldf = fstldf.rename(columns={"Type": "MACB"})
#
#            fstldf['Type_'] = fstldf['Mode'].str.extract('^(.)')
#            fstldf['PrevType_'] = fstldf['Mode'].str.extract('^..(.)')
#            fstldf['Permissions_'] = fstldf['Mode'].str.extract('^...(.........)')
#
#            #TODO: This should be moved to simple()
#            if windows :
#                fstldf.drop(columns=['UID','GID','Mode','Permissions_'],inplace=True)
#                #fstldf['Mode']=fstldf['Mode'].str.replace('r.xr.xr.x','')
#
#            if dateindex :
#                fstldf = fstldf.set_index('Date')
#
#    # Harmonize ---------------------------------------------------------------
#    if harmonize :
#        if fstltype == "pandas_dataframe-fstl-mactime-raw" or fstltype == "pandas_dataframe-fstl-mactime-adj":
#            print("- Harmonizing to fstl standard")
#            fstldf = harmonize_mactime(fstldf)
#
#    print("- Done.")
#
#    return fstldf

#def read_fstl_gui(rootpath="", notebook_file="", compname="", windows=False, dateindex=True):
#    """
#    Read a mactime filesystem timeline, either from saved pickle (if it has been 
#    generated before) or from a csv file
#    
#    If the current nobebook path is passed as a parameter, the json file location
#    will be searched for in the notebook (cell with definition of the f2read var.)
#    
#    Otherwise the user will be shown a widget to select the json file from disk
#    """
#
#    nbsave_prefix = compname+'_fstl_csv'
#
#    if notebook_file != "":
#        pattern = nbsave_prefix+'_f2read = "/.*$'
#
#        print("- Searching notebook for saved json file for this computer ("+nbsave_prefix+")")
#        hits = d4utl.nbgrep(notebook_file, pattern)
#        if hits:
#            f2read = hits[0].split(" = ")[1].strip('"')
#            print("  + Found: "+f2read)
#        else:
#            f2read = ""
#            print("  + Not Found")
#        print("")
#    else:
#        f2read = ""
#        
#    d4gui.file_select(rootpath, read_fstl, f2read=f2read, nbsave_prefix=nbsave_prefix)


# HARMONIZATION FUNCTIONS #####################################################

def harmonize(df, **kwargs):
    """ Convert DF in HAM format

        Args: 
            df (pandas.DataFrame): DF to harmonize
            kwargs(dict): harmonize options
        Returns: 
            pandas.DataFrame in HAM Format
    """
    objtype = d4com.data_identify(df)

    if objtype == "pandas_dataframe-mactime-raw":
        # Specific Harmonization Pre-Processing -----------------------------------
        df = df.rename(columns={"Type": "MACB"})

        df['Type_']        = df['Mode'].str.extract('^(.)')
        df['PrevType_']    = df['Mode'].str.extract('^..(.)')
        df['Permissions_'] = df['Mode'].str.extract('^...(.........)')

        # Deleted / Reallocated
        df['Deleted_']     = df['File Name'].str.contains(r'\ \(deleted\)$|\ \(deleted-reallocated\)$')
        df['Reallocated_'] = df['File Name'].str.contains(r'\ \(deleted-reallocated\)$')

        # [FT] Tag -> Tag_ | DriveLetter_ | VSS_ | EVOName_ | EvidenceName_ | Partition_ | FilePath_
        # FT
        if re.search(r'^[A-Z]\[vss[0-9][0-9]\]{.*}:', df['File Name'].iloc[0]):
            fncolsdf  = df['File Name'].str.split(":", 1, expand=True).rename(columns={0: "Tag_", 1: "FilePath_"})
            fncolsdf['FilePath-Hash_'] = fncolsdf['FilePath_'].str.lower().apply(hash)
            fncolsdf['FSType_']   = '-'
            df['Hostname_']    = '-'
            df['SHA256_Hash_'] = '-'

            fncols2df = fncolsdf['Tag_'].str.extract(r'([A-Z])\[vss(.*)\]{(.*)}', expand=True).rename(columns={0: "DriveLetter_", 1: "VSS_", 2: "EVOName_"})
            fncols2df['VSS_'] = fncols2df['VSS_'].astype(int)

            fncols3df = fncols2df['EVOName_'].str.extract('(.*)-ft-p(.*)', expand=True).rename(columns={0: "EvidenceName_", 1: "Partition_"})
            fncols3df['Partition_'] = fncols3df['Partition_'].astype(int)

            df = pd.concat([df, fncols2df, fncols3df, fncolsdf], axis=1)

        else:
            fncolsdf  = df['File Name'].str.split(":", 1, expand=True).rename(columns={0: "Tag_", 1: "FilePath_"})
            df = pd.concat([df, fncolsdf], axis=1)
            df['Hostname_']     = '-'
            df['EVOName_']      = '-'
            df['EvidenceName_'] = '-'
            df['Partition_']    = '-'
            df['FSType_']       = '-'
            df['DriveLetter_']  = '-'
            df['VSS_']          = '-'
            df['TSNTFSAttr_']   = '-'
            df['SHA256_Hash_']  = '-'

        # Deal with "($FILE_NAME)" string
        tsntfsattrmap = {True: 'FILE_NAME', False: 'STD_INFO'}
        df['TSNTFSAttr_']  = df['FilePath_'].str.contains(r'\ \(\$FILE_NAME\)$').map(tsntfsattrmap)
        df['FilePath_']    = df['FilePath_'].str.replace(r'\ \(\$FILE_NAME\)$','')

        df['FilePath_'] = df['FilePath_'].str.replace(r'\ \(deleted\)$|\ \(deleted-reallocated\)$','')
        
        # Generic Harmonization ---------------------------------------------------
        df = d4com.harmonize_common(df, datatype="flist", **kwargs)

        # Specific Harmonization Post-Processing ----------------------------------

        return df

# CORE FUNCTIONS (simple, analysis, etc.) #####################################

# simple ======================================================================

def simple_func(df, *args, **kwargs):
    """ Reformat the input df so the data is presented to the analyst in the
        friendliest possible way

    Parameters:
    df  (pd.dataframe):  Input data 
    
    Returns:
    pd.DataFrame: Optionally it will return the filtered dataframe, 
                  only if ret=True is set, constant & hidden columns included
                  If ret_out=True is set, then the output just as it is shown
                  (without constant/hidden columns) will be return
    """

    if d4.debug >= 3:
        print("DEBUG: [mctm] [simple_func()]")

    windows = kwargs.get('windows', True) 

    # Variables ----------------------------------------------------------------
    hiddencols =  ['File_Name', 'FilePath-Hash_', 'SHA256_Hash_']

    if windows :
        nonwincols = ['UID', 'GID', 'Mode', 'Permissions_']
        hiddencols = hiddencols + nonwincols

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Call to simple_common ----------------------------------------------------
    return d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

# def simple_func(df, *args, **kwargs):
#     """ Reformat the input df so the data is presented to the analyst in the
#         friendliest possible way
# 
#     Parameters:
#     df  (pd.dataframe):  Input data 
#     
#     Returns:
#     pd.DataFrame: Optionally it will return the filtered dataframe, 
#                   only if ret=True is set, constant & hidden columns included
#                   If ret_out=True is set, then the output just as it is shown
#                   (without constant/hidden columns) will be return
#     """
# 
#     # Arg. Parsing -------------------------------------------------------------
#     collapse_constant_cols = kwargs.get('collapse_constant_cols', True)
#     hide_cols              = kwargs.get('hide_cols', True)
#     # Show whatever selected on the screen or don't show anything   
#     out                    = kwargs.get('out', True)
#     # Show the resulting DF on the screen
#     out_df                 = kwargs.get('out_df', True)
#     # Return the resulting DataFrame
#     ret                    = kwargs.get('ret', False)
#     # Return the resulting DataFrame as shown in the screen
#     # (without hidden cols, collapsed cols, etc.; not in full)
#     ret_out                = kwargs.get('ret_out', False)
# 
#     # Check if out is set (will be needed later)
#     outargset = 'out' in kwargs.keys()
# 
#     # ret_out implies ret
#     if 'ret_out' in kwargs.keys() and ret_out:
#         ret = True
# 
#     # ret implies no out, unless out is set specifically
#     if ('ret' in kwargs.keys() or 'ret_out' in kwargs.keys()) and not outargset and ret:
#         # If ret is set to True by the user we will not provide stdout output,
#         # unless the user has specifically set out = True
#         out = False
# 
#     # Health Check -------------------------------------------------------------
#     dfnrows = len(df)
# 
#     if dfnrows == 0:
#         print("ERROR: Empty DataFrame.")
#         return
# 
#     # Var. Init. ---------------------------------------------------------------
#     # hiddencolsdf - Not used for now
#     hiddencolsdf = pd.DataFrame([])
# 
#     # Processing ---------------------------------------------------------------
#     if len(df.query('FilePath_.str.contains("^/Windows/System32$", case=False)', engine="python")) >= 1:
#         windows = True
#     else:
#         windows = False
# 
#     # TODO: This is mactime specific. Should probably move it somewhere else
#     if windows :
#         df = df.drop(columns=['UID','GID'])
# 
#     # "Beautification" ---------------------------------------------------------
#     dfb = df
# 
#     # standard - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     if hide_cols :
#         if 'FilePath-Hash_' in dfb.columns:
#             hiddencolsdf = pd.concat([hiddencolsdf, pd.DataFrame(['FilePath-Hash_'])], ignore_index=True)
#             dfb = dfb.drop(columns=['FilePath-Hash_'])
# 
#         if 'SHA256_Hash_' in dfb.columns:
#             hiddencolsdf = pd.concat([hiddencolsdf, pd.DataFrame(['SHA256_Hash_'])], ignore_index=True)
#             dfb = dfb.drop(columns=['SHA256_Hash_'])
# 
#     # mactime-specific - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     if collapse_constant_cols :
#         concolsdf, dfb = d4utl.collapse_constant_columns(dfb)
#     
#     # plaso-specific - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
#     # DISPLAY ==========================================================
#     nhiddencols = len(hiddencolsdf)
# 
#     if out :
#         print("")
#         display(Markdown("**Statistics:**\n<br>No. Entries: "+str(dfnrows)))
# 
#         if collapse_constant_cols :
#             show_constant_cols = True
#         else:
#             show_constant_cols = False
# 
#         if hide_cols  and nhiddencols != 0:
#             show_hidden_cols = True
#         else:
#             show_hidden_cols = False
# 
#         if show_constant_cols  and show_hidden_cols :
#             d4utl.display_side_by_side([hiddencolsdf, concolsdf], ['HIDDEN COLUMNS', 'CONSTANT COLUMNS'])
#         elif show_constant_cols  and show_hidden_cols == False:
#             display(Markdown("**Constant Columns**"))
#             display(concolsdf)
#         elif show_constant_cols == False and show_hidden_cols :
#             display(Markdown("**Hidden Columns**"))
#             display(hiddencolsdf)
# 
#         if out_df :
#             if dfnrows == 1:
#                 display(dfb.T)
#             else:
#                 display(dfb)
# 
#     # RETURN  ==========================================================
#     if ret :
#         if ret_out :
#             return dfb
#         else:
#             return df

# DATAFRAME ACCESSOR ##########################################################

# TODO: Consolidate the 2 accessors

@pd.api.extensions.register_dataframe_accessor("d4mctm")
class Ds4n6MctmAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_mactime")
class Ds4n6MactimeAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

# ANALYSIS ####################################################################

# analysis() function =========================================================
def analysis(*args, **kwargs):
    """ Redirects execution to analysis_func()
    """
    return analysis_func(*args, **kwargs)

def analysis_func(*args, **kwargs):
    # TODO: This should be substituted by a matrix of supported analysis types
    #       per data type, and then we can apply that matrix both to the 
    #       d4list function, as to the function call redirections at the end
    """ Umbrella function that redirects to different types of analysis 
        available on the input data

    Parameters:
    obj:          Input data (typically DF or dict of DFs)
    
    Returns:
    pd.DataFrame: Refer to each specific analysis function
    """

    def syntax():
        print('Syntax: analysis(obj, "analysis_type")\n')
        d4list("str-help")
        return

    def d4list(objtype):

        # Analysis Modules Available for this objective
        # anlav = False
        print("Available fstl analysis types:")
        print("- No analysis functions defined yet.")
        return

        # TEMPLATE
        #if objtype == "str-help" or objtype == "str-list" or  re.search("^pandas_dataframe-fstl-mactime-standard", objtype):
        #    anlav = True
        #    print("- XXXXXXXXXX:  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX (Input: fstldf)")

        # if anlav == False:
        #     print('- No analysis modules available for this object ('+objtype+').')

    nargs = len(args)

    if nargs == 0:
        syntax()
        return

    obj = args[0]

    objtype = d4com.data_identify(obj)

    if isinstance(obj, str):
        if obj == "list":
            d4list(objtype)
            return
        if obj == "help":
            syntax()
            return

    if nargs == 1:
        syntax()
        return

    anltype = args[1]

    if not isinstance(anltype, str):
        syntax()
        return

    if anltype == "help":
        syntax()
        return
    elif anltype == "list":
        d4list(objtype)
        return

    # TEMPLATE
    # If object is a dict of dfs
    #elif re.search("^pandas_dataframe-evtx_file_df", objtype):
    #    if anltype == "XXXXXXXXXXX":
    #        return XXXXXXXXXXXXXXXXXXXXX(*args, **kwargs)
    #else:
    #    print("ERROR: [fstl] Unsupported input data.")
    #    return

