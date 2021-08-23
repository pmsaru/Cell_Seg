import streamlit as st
import os
import cv2
import glob
import image_slicer
from tqdm import tqdm
from stardist.models import StarDist2D 
from csbdeep.utils import normalize
from skimage.measure import label, regionprops, regionprops_table
from scipy import ndimage
from stardist.plot import render_label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re 
import seaborn as sns

def im_upload():
	uploaded_file = st.file_uploader("Choose a file")
	if uploaded_file is not None:
		file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
		opencv_image = cv2.imdecode(file_bytes, 1)
		# fname = uploaded_file.name
		# plt.imsave(fname,opencv_image)
		st.image(opencv_image, channels="BGR")
		return uploaded_file

def bwareaopen(lbs,fname):
    if re.match('Leuko*',fname):
        th = 250
    else:
        th = 2000
    Z = lbs
    Zlabeled,Nlabels = ndimage.measurements.label(Z)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    for label,size in enumerate(label_size):
        if size > th:
            Z[Zlabeled == label] = 0
    return Z

def create_tiles(dp):
    num_tiles = 400
    tiles = image_slicer.slice(dp, num_tiles,save=False)
    return tiles

def model_star(tiles,fname):
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    label_im = []
    ori_im = []
    for i in tqdm(range(len(tiles))):
        im = tiles[i].image
        r,g,b = im.split()
        alpha = 1
        beta = -1
        if re.match('Leuko*',fname):
            ch = r
        else:
            ch = g
        img = cv2.convertScaleAbs(np.array(ch), alpha=alpha, beta=beta)
        lbl, _ = model.predict_instances(normalize(img))
        lbl = bwareaopen(lbl,fname)
        label_im.append(lbl)
        ori_im.append(img)
    return label_im,ori_im

def merge_tiles(tiles,label_im,ori_im,fname):
    k = len(tiles)
    row_ind=np.int64(tiles[k-1].position[1])
    col_ind=np.int64(tiles[k-1].position[0])
    lst=[]
    im_lst=[]
    cnt=0
    for i in range(row_ind):
        lst.append(np.hstack(label_im[cnt:cnt+col_ind]))
        im_lst.append(np.hstack(ori_im[cnt:cnt+col_ind]))
        cnt=cnt+col_ind
    final=np.vstack(lst[0:row_ind])  
    im_final=np.vstack(im_lst[0:row_ind])
    out=render_label(final, im_final, alpha = 1, alpha_boundary =0.8)
    st.image(out, channels="BGR")
    # plt.imsave(outputfile + fname + '.jpg',out)
    return True
    
def region_prop(label_im,ori_im,outputfile,fname):
    df= pd.DataFrame()
    rg_mn=[]
    for i in range(len(label_im)):
        props = regionprops_table(label_im[i], ori_im[i], properties=('eccentricity','major_axis_length',
                                                                     'minor_axis_length','mean_intensity'))
        df_tmp=pd.DataFrame(props)
        df_tmp["area"]=np.pi*df_tmp["major_axis_length"]*df_tmp["minor_axis_length"]
        frames = [df,df_tmp]
        df = pd.concat(frames)
    df.to_csv(outputfile + fname + '.csv',index=False)
    return True

def vplot(outputfile,fname):
    
    df = pd.read_csv(outputfile + fname + '.csv')
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0, 0])
    sns.violinplot(data=df["area"])
    ax.set_xlabel("Area")

    ax = fig.add_subplot(gs[0, 1])
    sns.violinplot(data=df["eccentricity"])
    ax.set_xlabel("Roundness")    

    ax = fig.add_subplot(gs[0, 2])
    sns.violinplot(data=df["mean_intensity"])
    ax.set_xlabel("Regional Mean Intensity")

    fig.tight_layout()
    plt.savefig(outputfile + fname + '_plot.jpg')
    return True

def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Upload and Segment','Cell Segmentation', 'Video', 'Face Detection', 'Feature Detection', 'Object Detection')
    )

    if selected_box == 'Welcome':
    	st.write("Hello Welcome") 
    if selected_box == 'Image Upload and Segment':
        filename=None
        filename=im_upload() 
        # st.write(filename.name)
        fname = filename.name.split('.')[0]
        # st.write(filename)
        tiles = create_tiles(filename.name)
        # st.write(tiles[0].image.size)
        # st.write(len(tiles))
        with st.spinner('Wait for it...'):
            label_im,ori_im = model_star(tiles,fname)    
            merge_tiles(tiles,label_im,ori_im,fname) 
        st.success("Done")  
        # print(f'Done merging tiles of {fname}') 
        # region_prop(label_im,ori_im,outputfile,fname)
        # print(f'Done with extracting features of {fname}') 
        # vplot(outputfile,fname)
    # if selected_box == 'Video':
    # 	video()
    # if selected_box == 'Face Detection':
    # 	face_detection()
    # if selected_box == 'Feature Detection':
    # 	feature_detection()
    # if selected_box == 'Object Detection':
    # 	object_detection()

if __name__ == "__main__":
	main()
