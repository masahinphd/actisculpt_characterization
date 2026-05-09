import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

st.set_page_config(layout="wide", page_title="Active Flow Sculpting Characterization", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Active Flow Sculpting Particle Characterization</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-top: 0; font-size: 1.05rem;'>Performs Mixing Coefficient, Moment of Inertia, Fluid Displacement, and Uniformity analyses natively.</p>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        justify-content: center;
    }
    [data-testid="stTab"] {
        font-size: 1.05rem;
        font-weight: 700;
    }
    [data-testid="stTab"] p {
        font-size: 1.05rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def parse_amp_str(amp_str_val):
    try:
        parsed_amps = [int(float(a.strip())) for a in amp_str_val.strip("()[]").split(",")]
        if len(parsed_amps) == 3 and "(" in amp_str_val:
            return list(range(parsed_amps[0], parsed_amps[2] + 1, parsed_amps[1]))
        return parsed_amps
    except ValueError:
        pass
    return []

def render_tab_header(prefix, allow_separate_amps=False, default_freq_mask="1111"):
    st.subheader("Filter, Amplitudes & Image Config")
    col1, col2, col3 = st.columns([1, 2.5, 1.2])
    def def_amps(f):
        if prefix == "mix": return "20, 40, 60, 80, 100, 120, 200"
        if prefix == "fd":
            fd_defaults = {
                1: "(10,5,70)",
                2: "(10,5,50)",
                3: "(10,5,60)",
                4: "(15,10,110)",
            }
            return fd_defaults.get(int(f), "(10,5,50)")
        if prefix == "uni": return "(20,20,120)"
        if prefix == "moi": return "(10,5,160)"
        return "(15,10,90)" if f == 4 else "(10,5,200)"
    amp_label = "Discrete Amplitudes List" if prefix == "mix" else "Amplitudes (initial, step, last)"
    
    with col1:
        freq_str = st.text_input("IDT Selection Mask (1=On, 0=Off)", default_freq_mask, key=f"{prefix}_freq_str")
        clean_str = freq_str.replace(",", "").replace(" ", "")
        raw_freqs = list(clean_str)
        if all(v in ["0", "1"] for v in raw_freqs):
            frequency_values = [i + 1 for i, v in enumerate(raw_freqs) if v == "1"]
        else:
            frequency_values = [int(v) for v in raw_freqs if v.isdigit()]
            
    with col2:
        if allow_separate_amps and len(frequency_values) > 1:
            use_sep_amps = st.checkbox("Separate Amplitudes per IDT", value=True, key=f"{prefix}_use_sep_amps")
            sep_amps = {}
            if use_sep_amps:
                cols = st.columns(len(frequency_values))
                for idx, f in enumerate(frequency_values):
                    with cols[idx]:
                        sep_amps[f] = st.text_input(f"IDT {f} Amps", value=def_amps(f), key=f"{prefix}_fd_a_{f}")
            else:
                a_str = st.text_input(amp_label, def_amps(1), key=f"{prefix}_fd_amp_all")
        else:
            if len(frequency_values) > 1:
                a_str = st.text_input(amp_label, def_amps(1), key=f"{prefix}_fd_amp_all")
            else:
                f_base = frequency_values[0] if frequency_values else 1
                a_str = st.text_input(amp_label, def_amps(f_base), key=f"{prefix}_fd_amp_base")
            
    with col3:
        crop_images = st.checkbox("Crop Images", value=True, key=f"{prefix}_crop")
        colA, colB = st.columns(2)
        with colA:
            frame_length = st.number_input("Frame Length", value=174, min_value=50, step=2, key=f"{prefix}_frame")
        with colB:
            pixel_size_mm = st.number_input("Pixel Size (mm)", value=(1/0.1928*10**-3), format="%.6f", step=0.0001, key=f"{prefix}_pixel")

    st.markdown("---")

    st.subheader("Data Paths")
    col6, col7 = st.columns(2)
    default_jpath = "./Mixing-Uniformity-MoI/Mixing-Uniformity-MoI.json" if prefix in ["mix", "uni", "moi"] else "./Fluid Dislocation/Common_Results.json"
    default_ipath = "./Mixing-Uniformity-MoI" if prefix in ["mix", "uni", "moi"] else "./Fluid Dislocation"
    with col6:
        j_path = st.text_input("JSON Results Path", default_jpath, key=f"{prefix}_jpath")
    with col7:
        i_path = st.text_input("Particles Base Directory", default_ipath, key=f"{prefix}_ipath")

    def tab_get_amp_range(f):
        if allow_separate_amps and len(frequency_values) > 1:
            if use_sep_amps:
                return parse_amp_str(sep_amps[f])
            return parse_amp_str(a_str)
        return parse_amp_str(a_str)

    return {
        'frequency_values': frequency_values,
        'crop_images': crop_images,
        'frame_length': frame_length,
        'pixel_size_mm': pixel_size_mm,
        'j_path': j_path,
        'i_path': i_path,
        'get_amp_range': tab_get_amp_range
    }

def load_data(data_dir):
    try:
        return pd.read_json(data_dir, orient='records')
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
        return None


def resolve_image_directory(images_base_dir, expname, metadata=None):
    filedir = os.path.join(images_base_dir, expname)
    if os.path.exists(filedir):
        return filedir

    try:
        known_dirs = [d for d in os.listdir(images_base_dir) if os.path.isdir(os.path.join(images_base_dir, d))]
    except Exception:
        return filedir

    expname_clean = expname.strip().replace(' ', '')
    candidates = []

    if '-' in expname_clean:
        prefix = expname_clean.rsplit('-', 1)[0]
        candidates.extend([d for d in known_dirs if d.replace(' ', '').startswith(prefix)])

    if metadata is not None and len(metadata) >= 6:
        try:
            f = str(metadata[0]).strip()
            amp = str(metadata[1]).strip()
            q = str(metadata[2]).strip()
            qt = str(metadata[3]).strip()
            sg = str(metadata[4]).strip()
            n = str(metadata[5]).strip()
            prefix = f"F{f}Amp{amp}Q{q}QT{qt}SG{sg}N{n}"
            candidates.extend([d for d in known_dirs if d.replace(' ', '').startswith(prefix.replace(' ', ''))])
        except Exception:
            pass

    if not candidates and '-' in expname_clean:
        prefix = expname_clean.split('-', 1)[0]
        candidates.extend([d for d in known_dirs if d.replace(' ', '').startswith(prefix)])

    if not candidates and metadata is not None and len(metadata) >= 2:
        prefix = f"F{metadata[0]}Amp{metadata[1]}"
        candidates.extend([d for d in known_dirs if d.replace(' ', '').startswith(prefix)])

    if candidates:
        candidate = sorted(set(candidates))[0]
        return os.path.join(images_base_dir, candidate)

    return filedir

def filter_data(df, amplitude_range, frequency_values, strict_valid=True):
    def check_valid(x):
        if not (isinstance(x, list) and len(x) > 1): return False
        if int(x[1]) not in amplitude_range or x[0] not in frequency_values: return False
        if x[2] != '11': return False
        if strict_valid:
            if len(x) <= 7 or x[7] != 'Valid': return False
        return True

    idx = df[df['MetaData'].apply(check_valid)].index.tolist()
    idx.sort(key=lambda x: (df['MetaData'][x][0], int(df['MetaData'][x][1])))
    
    # Handle 120 repetition as in the original script
    to_pop = []
    if len(idx) > 1:
        for i in range(len(idx) - 1):
            if df['MetaData'][idx[i]][0] == df['MetaData'][idx[i + 1]][0] and \
               int(df['MetaData'][idx[i]][1]) == 120 and \
               int(df['MetaData'][idx[i + 1]][1]) == 120:
                to_pop.append(i) # pop the primary 120 retaining the subsequent 120.0 dataset containing more samples
        for i in reversed(to_pop):
            idx.pop(i)
    return idx

def process_frame(img, length, crop):
    # Make even
    if img.shape[0] % 2 != 0: img = img[:-1, :]
    if img.shape[1] % 2 != 0: img = img[:, :-1]
    
    if crop:
        frame = np.zeros((length, length, 3), dtype=np.uint8)
        if length <= img.shape[0] and length <= img.shape[1]:
            frame = img[(img.shape[0]//2-length//2):(length//2+img.shape[0]//2), 
                        (img.shape[1]//2-length//2):(length//2+img.shape[1]//2), :]
            img = frame
        elif length <= img.shape[0] and length >= img.shape[1]:
            frame = img[(img.shape[0]//2-length//2):(length//2+img.shape[0]//2), 
                        (length//2-img.shape[1]//2):(length//2+img.shape[1]//2), :] 
            img = frame
        elif length >= img.shape[0] and length <= img.shape[1]:
            frame = img[(length//2-img.shape[0]//2):(length//2+img.shape[0]//2), 
                        (img.shape[1]//2-length//2):(length//2+img.shape[1]//2), :]
            img = frame
        else:    
            frame[(length//2-img.shape[0]//2):(length//2+img.shape[0]//2), 
                  (length//2-img.shape[1]//2):(length//2+img.shape[1]//2), :] = img
            img = frame
    return img

def binarize_image(img):
    img_r = np.clip(cv.subtract(img[:, :, 0], img[:, :, 2]), 0, 255)
    img_b = np.clip(cv.subtract(img[:, :, 2], img[:, :, 0]), 0, 255)
    img_r[img_r > 0] = 1
    img_b[img_b > 0] = 1
    return img_r, img_b

def execute_analysis(analysis_type, tc, use_own_center=False, fd_idt1_scale=1.0, fd_idt2_scale=1.4, fd_idt3_scale=1.0, fd_idt4_scale=1.65, custom_amps=None, use_voltage=False, conversion_rate=2.0, strict_uni_calc=False, use_path_fit=False, show_stacked_evolution=False, show_std_dev=False, show_moi_channel_separated=False):
    frequency_values = tc['frequency_values']
    crop_images = tc['crop_images']
    frame_length = tc['frame_length']
    pixel_size_mm = tc['pixel_size_mm']
    j_path = tc['j_path']
    i_path = tc['i_path']
    get_amp_range = tc['get_amp_range']
    
    st.info("Loading data and images...")
    
    total_idx_count = 0
    df_dict = {}
    idx_dict = {}
    
    for f in frequency_values:
        if analysis_type == 'mixing':
            if custom_amps:
                a_range = parse_amp_str(custom_amps)
            else:
                a_range = get_amp_range(f)
        else:
            a_range = get_amp_range(f)
            
        if not a_range: continue
        
        df = load_data(j_path)
        if df is not None:
            is_mixing = (analysis_type == 'mixing')
            idx = filter_data(df, a_range, [f], strict_valid=not is_mixing)
            if len(idx) > 0:
                df_dict[f] = df
                idx_dict[f] = idx
                total_idx_count += len(idx)
                
    if total_idx_count == 0:
        st.warning("No data found for the selected amplitude and frequency range.")
        return
        
    all_amp_vals = np.zeros(total_idx_count)
    all_frequencies = np.zeros(total_idx_count)
    
    # Mixing
    mixing_results_act_r = []
    mixing_results_act_b = []
    mixing_results_noact_r = []
    mixing_results_noact_b = []
    
    # Uniformity
    stability_r = np.zeros(total_idx_count)
    stability_b = np.zeros(total_idx_count)
    stability_r_std = np.zeros(total_idx_count)
    stability_b_std = np.zeros(total_idx_count)
    
    # MOI + Fluid Displacement
    center_r_avg = np.zeros((total_idx_count, 2))
    center_b_avg = np.zeros((total_idx_count, 2))
    noact_center_r_avg = np.zeros((total_idx_count, 2))
    noact_center_b_avg = np.zeros((total_idx_count, 2))
    M_r_avg = np.zeros((total_idx_count, 3, 2))
    M_b_avg = np.zeros((total_idx_count, 3, 2))
    M_r_std = np.zeros((total_idx_count, 3, 2))
    M_b_std = np.zeros((total_idx_count, 3, 2))
    
    # Per-particle data for standard deviation tracking
    all_center_r_per_particle = []  # List of lists: [amplitude_idx][particle_idx]
    all_center_b_per_particle = []
    all_noact_center_r_per_particle = []
    all_noact_center_b_per_particle = []

    all_bg_images = []
    all_act_red_masks = []

    progress_bar = st.progress(0)
    
    j = 0
    for f in frequency_values:
        if f not in df_dict: continue
        df = df_dict[f]
        idx = idx_dict[f]
        images_base_dir = i_path
        
        for experiment_idx in idx:
            raw_expname = df['Name'][experiment_idx]
            if isinstance(raw_expname, str) and raw_expname.startswith("["):
                import ast
                try:
                    expname = ast.literal_eval(raw_expname)[0]
                except:
                    expname = raw_expname.strip("[]'\"")
            elif isinstance(raw_expname, list):
                expname = raw_expname[0]
            else:
                expname = str(raw_expname)
                
            all_amp_vals[j] = df['MetaData'][experiment_idx][1]
            all_frequencies[j] = df['MetaData'][experiment_idx][0]
            
            filedir = resolve_image_directory(images_base_dir, expname, df['MetaData'][experiment_idx])
            
            images = []
            if os.path.exists(filedir):
                images = sorted([img for img in glob.glob(os.path.join(filedir, '*.tif')) if '00.' not in img], key=lambda x: os.path.basename(x)[0])
            else:
                st.warning(f"Directory not found: {filedir}")
                continue
                
            act_images = [img for img in images if 'NoACT' not in img and 'Excluded' not in img]
            noact_images = [img for img in images if 'NoACT' in img]
            
            first_img_path = noact_images[0] if len(noact_images) > 0 else (act_images[0] if len(act_images) > 0 else None)
            bg_image_arr = np.zeros((frame_length, frame_length, 3), dtype=np.uint8)
            if first_img_path is not None:
                first_img = cv.imread(first_img_path)
                if first_img is not None:
                    first_img = cv.cvtColor(first_img, cv.COLOR_BGR2RGB)
                    bg_image_arr = process_frame(first_img, frame_length, crop_images)
            all_bg_images.append(bg_image_arr)
            
            red_mask_arr = np.zeros((frame_length, frame_length), dtype=np.uint8)
            if len(act_images) > 0:
                act_img = cv.imread(act_images[0])
                if act_img is not None:
                    act_img = cv.cvtColor(act_img, cv.COLOR_BGR2RGB)
                    act_img = process_frame(act_img, frame_length, crop_images)
                    img_r, _ = binarize_image(act_img)
                    red_mask_arr = img_r
            all_act_red_masks.append(red_mask_arr)

            cum_images_r = np.zeros((frame_length, frame_length, 1), dtype=np.uint16)
            cum_images_b = np.zeros((frame_length, frame_length, 1), dtype=np.uint16)
            
            act_std_r, act_std_b = [], []
            noact_std_r, noact_std_b = [], []
            
            act_center_r, act_center_b = [], []
            noact_center_r, noact_center_b = [], []
            
            act_M_r, act_M_b = [], []
            noact_M_r, noact_M_b = [], []
            
            pixel_area = pixel_size_mm**2  # Area of one pixel in physical units (mm²)
            
            def process_image_list(img_list, is_act=True):
                for i_path in img_list:
                    img = cv.imread(i_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = process_frame(img, frame_length, crop_images)
                    img_r, img_b = binarize_image(img)
                    
                    if analysis_type in ['mixing', 'uniformity']:
                        if is_act:
                            if analysis_type == 'mixing':
                                raw_r = img[:,:,0] / np.max(img[:,:,0]) if np.max(img[:,:,0]) > 0 else img[:,:,0] * 0.0
                                raw_b = img[:,:,2] / np.max(img[:,:,2]) if np.max(img[:,:,2]) > 0 else img[:,:,2] * 0.0
                                act_std_r.append(np.std(raw_r))
                                act_std_b.append(np.std(raw_b))
                            if analysis_type == 'uniformity':
                                cum_images_r[:,:,0] = cum_images_r[:,:,0] + img_r
                                cum_images_b[:,:,0] = cum_images_b[:,:,0] + img_b
                                # store raw active pixel counts per frame for std
                                act_std_r.append(float(np.sum(img_r > 0)))
                                act_std_b.append(float(np.sum(img_b > 0)))
                        else:
                            if analysis_type == 'mixing':
                                raw_r = img[:,:,0] / np.max(img[:,:,0]) if np.max(img[:,:,0]) > 0 else img[:,:,0] * 0.0
                                raw_b = img[:,:,2] / np.max(img[:,:,2]) if np.max(img[:,:,2]) > 0 else img[:,:,2] * 0.0
                                noact_std_r.append(np.std(raw_r))
                                noact_std_b.append(np.std(raw_b))
                        
                    if analysis_type in ['fluid_dislocation', 'moi']:
                        M_r_moments = cv.moments(img_r)
                        M_b_moments = cv.moments(img_b)
                        
                        cx_r = int(M_r_moments["m10"]/M_r_moments["m00"]) if (use_own_center and M_r_moments["m00"]!=0) else img.shape[1]//2
                        cy_r = int(M_r_moments["m01"]/M_r_moments["m00"]) if (use_own_center and M_r_moments["m00"]!=0) else img.shape[0]//2
                        
                        cx_b = int(M_b_moments["m10"]/M_b_moments["m00"]) if (use_own_center and M_b_moments["m00"]!=0) else img.shape[1]//2
                        cy_b = int(M_b_moments["m01"]/M_b_moments["m00"]) if (use_own_center and M_b_moments["m00"]!=0) else img.shape[0]//2
                        
                        if analysis_type == 'moi':
                            y_indices, x_indices = np.indices(img_r.shape)
                            dist_x_r, dist_y_r = np.abs(x_indices - cx_r), np.abs(y_indices - cy_r)
                            dist_z_r = np.hypot(dist_x_r, dist_y_r)
                            
                            dist_x_b, dist_y_b = np.abs(x_indices - cx_b), np.abs(y_indices - cy_b)
                            dist_z_b = np.hypot(dist_x_b, dist_y_b)
                            
                            # Moment of inertia: I = Σ mask(x,y) * (distance_physical(x,y))^2 * pixel_area
                            # where distance_physical = distance_pixel * pixel_size_mm
                            val_Mr = [np.sum(img_r * (dist_y_r * pixel_size_mm)**2) * pixel_area, 
                                      np.sum(img_r * (dist_x_r * pixel_size_mm)**2) * pixel_area, 
                                      np.sum(img_r * (dist_z_r * pixel_size_mm)**2) * pixel_area]
                            val_Mb = [np.sum(img_b * (dist_y_b * pixel_size_mm)**2) * pixel_area, 
                                      np.sum(img_b * (dist_x_b * pixel_size_mm)**2) * pixel_area, 
                                      np.sum(img_b * (dist_z_b * pixel_size_mm)**2) * pixel_area]

                        if is_act:
                            act_center_r.append([cx_r, cy_r])
                            act_center_b.append([cx_b, cy_b])
                            if analysis_type == 'moi':
                                act_M_r.append(val_Mr)
                                act_M_b.append(val_Mb)
                        else:
                            noact_center_r.append([cx_r, cy_r])
                            noact_center_b.append([cx_b, cy_b])
                            if analysis_type == 'moi':
                                noact_M_r.append(val_Mr)
                                noact_M_b.append(val_Mb)

            process_image_list(noact_images, is_act=False)
            process_image_list(act_images, is_act=True)
            
            if analysis_type == 'mixing':
                mixing_results_act_r.append(np.array(act_std_r))
                mixing_results_act_b.append(np.array(act_std_b))
                mixing_results_noact_r.append(np.array(noact_std_r))
                mixing_results_noact_b.append(np.array(noact_std_b))
            
            if analysis_type == 'uniformity':
                if len(act_images) > 0:
                    if strict_uni_calc:
                        stability_r[j] = np.sum(cum_images_r[:,:,0] == len(act_images)) / (np.sum(cum_images_r[:,:,0] > 0) + 1e-10)
                        stability_b[j] = np.sum(cum_images_b[:,:,0] == len(act_images)) / (np.sum(cum_images_b[:,:,0] > 0) + 1e-10)
                    else:
                        stability_r[j] = np.sum(cum_images_r[:,:,0] >= len(act_images)//2) / (np.sum(cum_images_r[:,:,0] > 0) + 1e-10)
                        stability_b[j] = np.sum(cum_images_b[:,:,0] >= len(act_images)//2) / (np.sum(cum_images_b[:,:,0] > 0) + 1e-10)
                    # normalized std: CV of per-frame active area (as fraction of bar height)
                    if len(act_std_r) > 1:
                        arr_r = np.array(act_std_r, dtype=float)
                        arr_b = np.array(act_std_b, dtype=float)
                        # normalize each to 0-1 by max active area in this experiment
                        max_r = np.max(arr_r) if np.max(arr_r) > 0 else 1.0
                        max_b = np.max(arr_b) if np.max(arr_b) > 0 else 1.0
                        norm_r = arr_r / max_r
                        norm_b = arr_b / max_b
                        cv_r = np.std(norm_r) / (np.mean(norm_r) + 1e-10)
                        cv_b = np.std(norm_b) / (np.mean(norm_b) + 1e-10)
                        # store as fraction of bar (will be multiplied by bar height in plot)
                        stability_r_std[j] = cv_r
                        stability_b_std[j] = cv_b
            
            if analysis_type in ['fluid_dislocation', 'moi']:
                if len(act_M_r) > 0 and analysis_type == 'moi':
                    M_r_avg[j, :, 1] = np.mean(act_M_r, axis=0)
                    M_r_std[j, :, 1] = np.std(act_M_r, axis=0)
                    M_b_avg[j, :, 1] = np.mean(act_M_b, axis=0)
                    M_b_std[j, :, 1] = np.std(act_M_b, axis=0)
                if len(noact_M_r) > 0 and analysis_type == 'moi':
                    M_r_avg[j, :, 0] = np.mean(noact_M_r, axis=0)
                    M_r_std[j, :, 0] = np.std(noact_M_r, axis=0)
                    M_b_avg[j, :, 0] = np.mean(noact_M_b, axis=0)
                    M_b_std[j, :, 0] = np.std(noact_M_b, axis=0)

                if len(act_center_r) > 0:
                    center_r_avg[j, :] = np.mean(act_center_r, axis=0)
                    center_b_avg[j, :] = np.mean(act_center_b, axis=0)
                    # Store per-particle data for std dev calculation
                    act_center_r_flipy = np.array(act_center_r)
                    act_center_r_flipy[:, 1] = frame_length - act_center_r_flipy[:, 1]
                    act_center_b_flipy = np.array(act_center_b)
                    act_center_b_flipy[:, 1] = frame_length - act_center_b_flipy[:, 1]
                    all_center_r_per_particle.append(act_center_r_flipy)
                    all_center_b_per_particle.append(act_center_b_flipy)
                    center_r_avg[j, 1] = frame_length - center_r_avg[j, 1]
                    center_b_avg[j, 1] = frame_length - center_b_avg[j, 1]

                if len(noact_center_r) > 0:
                    noact_center_r_avg[j, :] = np.mean(noact_center_r, axis=0)
                    noact_center_b_avg[j, :] = np.mean(noact_center_b, axis=0)
                    # Store per-particle noact data
                    noact_center_r_flipy = np.array(noact_center_r)
                    noact_center_r_flipy[:, 1] = frame_length - noact_center_r_flipy[:, 1]
                    noact_center_b_flipy = np.array(noact_center_b)
                    noact_center_b_flipy[:, 1] = frame_length - noact_center_b_flipy[:, 1]
                    all_noact_center_r_per_particle.append(noact_center_r_flipy)
                    all_noact_center_b_per_particle.append(noact_center_b_flipy)
                    noact_center_r_avg[j, 1] = frame_length - noact_center_r_avg[j, 1]
                    noact_center_b_avg[j, 1] = frame_length - noact_center_b_avg[j, 1]

            j += 1
            progress_bar.progress(j / total_idx_count)
        
    amp_diff = np.diff(all_amp_vals)
    amp_diff_idx = np.where(amp_diff < 0)[0]
    freq_amp_vals = np.split(all_amp_vals, amp_diff_idx + 1)
    idt_freqs = np.split(all_frequencies, amp_diff_idx + 1)
    
    if analysis_type == 'mixing':        
        mix_coeff_raw = []
        for i in range(total_idx_count):
            mix_r = mixing_results_act_r[i]
            mix_b = mixing_results_act_b[i]
            mix_raw = np.mean([mix_r, mix_b], axis=0)
            mix_coeff_raw.append(mix_raw)
            
        if mixing_results_noact_r and mixing_results_noact_b:
            flat_noact_r = np.concatenate([np.array(arr) for arr in mixing_results_noact_r])
            flat_noact_b = np.concatenate([np.array(arr) for arr in mixing_results_noact_b])
            noact_mix_raw = np.mean([flat_noact_r, flat_noact_b], axis=0)
        else:
            noact_mix_raw = []
        
        splits = [0] + list(amp_diff_idx + 1) + [len(mix_coeff_raw)]
        mix_raw_split = [mix_coeff_raw[splits[k]:splits[k+1]] for k in range(len(splits)-1)]

        unique_amps_set = set()
        for arr in freq_amp_vals:
            unique_amps_set.update(np.array(arr, dtype=float))
        unique_amps_list = sorted(list(unique_amps_set))
        if len(noact_mix_raw) > 0 and 0 not in unique_amps_list:
            unique_amps_list = [0] + unique_amps_list

        import matplotlib as mpl
        original_rc = mpl.rcParams.copy()
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.size': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'mathtext.fontset': 'custom',
            'mathtext.rm': 'Cambria Math',
            'mathtext.it': 'Cambria Math:italic',
            'mathtext.bf': 'Cambria Math:bold'
        })
        cm_to_in = 1 / 2.54
        fig, ax = plt.subplots(figsize=(7.5 * cm_to_in, 5 * cm_to_in))
        num_idts = len(freq_amp_vals)
        alphas = [1.0, 0.8, 0.6, 0.4]
        
        box_width = 0.8 / max(1, num_idts)
        
        if len(noact_mix_raw) > 0 and 0 in unique_amps_list:
            pos_index = unique_amps_list.index(0)
            ax.boxplot([noact_mix_raw], positions=[pos_index], widths=box_width, patch_artist=True,
                       boxprops=dict(facecolor="black", edgecolor='none'),
                       capprops=dict(color="black", linewidth=1),
                       whiskerprops=dict(color="black"),
                       medianprops=dict(linewidth=0), showfliers=False)

        amp_presence = {a: [] for a in unique_amps_list}
        for i in range(num_idts):
            for a in np.array(freq_amp_vals[i], dtype=float):
                amp_presence[a].append(i)

        for i in range(num_idts):
            current_amps = np.array(freq_amp_vals[i], dtype=float)
            positions = []
            
            for a in current_amps:
                present_idts = amp_presence[a]
                k = len(present_idts)
                rank = present_idts.index(i)
                offset = (rank - (k - 1) / 2) * box_width
                positions.append(unique_amps_list.index(a) + offset)
                
            alpha_val = alphas[i % len(alphas)]
            if len(positions) > 0:
                ax.boxplot(list(mix_raw_split[i]), positions=positions, widths=box_width*0.9, patch_artist=True,
                           boxprops=dict(facecolor="C1", alpha=alpha_val, edgecolor='none'),
                           capprops=dict(color="C1", alpha=alpha_val, linewidth=1),
                           whiskerprops=dict(color="C1", alpha=alpha_val),
                           medianprops=dict(linewidth=0), showfliers=False)
                ax.plot([], [], color="C1", alpha=alpha_val, marker='s', linestyle='', label=f'IDT {int(idt_freqs[i][0])}')

        ax.set_xticks(range(len(unique_amps_list)))
        if use_voltage:
            ax.set_xticklabels([f"{a * conversion_rate:g}" for a in unique_amps_list])
            ax.set_xlabel(r"Amplified voltage, $V_{out}$, V")
        else:
            ax.set_xticklabels([str(int(a)) for a in unique_amps_list])
            ax.set_xlabel(r"Acoustic amplitude, a.u.")
            
        ax.set_ylabel(r"Standard deviation, $\sigma$")
        ax.tick_params(direction='out')
        ax.set_ylim(-0.02, 0.55)
        
        # Add ideal indicators
        line_sep = ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        txt_sep = ax.text(x=len(unique_amps_list) - 0.55, y=0.505, s='Ideal separation', color='gray', ha='right', va='bottom')
        
        line_mix = ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        txt_mix = ax.text(x=len(unique_amps_list) - 0.55, y=0.005, s='Ideal mixing', color='gray', ha='right', va='bottom')

        ax.legend()
        
        if len(unique_amps_list) >= 2:
            idx_last = len(unique_amps_list) - 1
            idx_prev = len(unique_amps_list) - 2
            a_last = unique_amps_list[idx_last]
            a_prev = unique_amps_list[idx_prev]
            if a_last - a_prev > 40:
                d = .02
                break_x = (idx_prev + idx_last) / 2
                trans = ax.get_xaxis_transform()
                ax.plot([break_x - d*5, break_x + d*5], [-d, +d], color='k', transform=trans, clip_on=False, lw=1.5)
                ax.plot([break_x - d*5 + 0.15, break_x + d*5 + 0.15], [-d, +d], color='k', transform=trans, clip_on=False, lw=1.5)

        st.pyplot(fig)
        
        import io
        legend = ax.get_legend()
        if legend:
            legend.remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
        line_sep.remove()
        txt_sep.remove()
        line_mix.remove()
        txt_mix.remove()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, transparent=True, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="Download Clean Plot (Transparent, 300 DPI)",
            data=buf,
            file_name="mixing_coefficient_clean.png",
            mime="image/png"
        )
        plt.rcParams.update(original_rc)
    
    elif analysis_type == 'uniformity':
        original_rc = plt.rcParams.copy()
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 6})
        
        # Dimensions strictly from Code2_Uniformity_v2.ipynb
        fig, ax = plt.subplots(1, 1, figsize=(3.9/2.54, 4.85/2.54))
        
        stability = np.mean([stability_r, stability_b], axis=0)
        stability_std = np.mean([stability_r_std, stability_b_std], axis=0)

        # Overall average uniformity across all selected IDTs and amplitudes.
        finite_uniformity = stability[np.isfinite(stability)]
        if finite_uniformity.size > 0:
            st.info(f"Average Uniformity (all selected IDTs/amplitudes): {np.mean(finite_uniformity):.4f}")

        stab_split = np.split(stability, amp_diff_idx + 1)
        stab_std_split = np.split(stability_std, amp_diff_idx + 1)
        
        n_freqs = len(freq_amp_vals)
        
        unique_amps_set = set()
        for arr in freq_amp_vals:
            unique_amps_set.update(np.array(arr, dtype=float))
        unique_amps_list = sorted(list(unique_amps_set))
        
        amp_presence = {a: [] for a in unique_amps_list}
        for i in range(n_freqs):
            for a in np.array(freq_amp_vals[i], dtype=float):
                amp_presence[a].append(i)
                
        min_step = min(np.diff(unique_amps_list)) if len(unique_amps_list) > 1 else 20.0
        max_k = max(len(present) for present in amp_presence.values()) if amp_presence else 1
        base_width = (min_step * 0.8) / max_k
        if use_voltage:
            base_width = base_width * conversion_rate
            
        alphas = [1.0, 0.8, 0.6, 0.4, 0.2]

        for i in range(n_freqs):
            amps = np.array(freq_amp_vals[i], dtype=float)
            stabs = stab_split[i]
            stabs_std = stab_std_split[i]
            x_vals = []
            
            for a in amps:
                present_idts = amp_presence[a]
                k = len(present_idts)
                rank = present_idts.index(i)
                offset = (rank - (k - 1) / 2) * base_width
                
                x_center = a
                if use_voltage: x_center = a * conversion_rate
                x_vals.append(x_center + offset)
                
            ax.bar(x_vals, stabs, width=base_width, color='C1', alpha=alphas[i%len(alphas)], label=f'IDT {int(idt_freqs[i][0])}')
            # error bars: CV * bar_height so they scale proportionally
            import matplotlib.colors as mcolors
            c1_rgb = np.array(mcolors.to_rgb('C1'))
            bar_color = tuple(c1_rgb * 0.65)
            yerr_vals = stabs_std * stabs
            ax.errorbar(x_vals, stabs, yerr=yerr_vals, fmt='none', ecolor=bar_color, elinewidth=0.4, capsize=0.8, capthick=0.4)
            
        ax.tick_params(direction='out')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.2, step=0.25))
        
        if use_voltage:
            tick_positions = [a * conversion_rate for a in unique_amps_list]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{v:.4g}" for v in tick_positions])
            ax.set_xlim(min(tick_positions) - base_width * 2, max(tick_positions) + base_width * 2)
            ax.set_xlabel(r"Amplified voltage, $V_{out}$, V")
        else:
            ax.set_xticks(unique_amps_list)
            ax.set_xticklabels([str(int(a)) for a in unique_amps_list])
            ax.set_xlim(min(unique_amps_list) - base_width * 2, max(unique_amps_list) + base_width * 2)
            ax.set_xlabel('Acoustic Amplitude, a.u.')
            
        ax.set_ylabel('Uniformity, a.u.')
        
        st.pyplot(fig)
        
        import io
        legend = ax.get_legend()
        if legend:
            legend.remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=500, transparent=True, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="Download Clean Uniformity Plot",
            data=buf,
            file_name="uniformity_clean.png",
            mime="image/png"
        )
        plt.rcParams.update(original_rc)

    elif analysis_type == 'fluid_dislocation':
        center_r_split = np.split(center_r_avg, amp_diff_idx + 1)
        center_b_split = np.split(center_b_avg, amp_diff_idx + 1)
        noact_center_r_split = np.split(noact_center_r_avg, amp_diff_idx + 1)
        noact_center_b_split = np.split(noact_center_b_avg, amp_diff_idx + 1)
        
        # Split per-particle data if show_std_dev is enabled
        center_r_per_particle_split = None
        center_b_per_particle_split = None
        noact_center_r_per_particle_split = None
        noact_center_b_per_particle_split = None
        
        if show_std_dev:
            # Helper function to split list of arrays by indices (can't use np.split for inhomogeneous shapes)
            def split_list_by_indices(lst, indices):
                if not lst or len(lst) == 0:
                    return None
                result = []
                prev = 0
                for idx in indices:
                    result.append(lst[prev:idx])
                    prev = idx
                result.append(lst[prev:])
                return result
            
            center_r_per_particle_split = split_list_by_indices(all_center_r_per_particle, amp_diff_idx + 1) if len(all_center_r_per_particle) > 0 else None
            center_b_per_particle_split = split_list_by_indices(all_center_b_per_particle, amp_diff_idx + 1) if len(all_center_b_per_particle) > 0 else None
            noact_center_r_per_particle_split = split_list_by_indices(all_noact_center_r_per_particle, amp_diff_idx + 1) if len(all_noact_center_r_per_particle) > 0 else None
            noact_center_b_per_particle_split = split_list_by_indices(all_noact_center_b_per_particle, amp_diff_idx + 1) if len(all_noact_center_b_per_particle) > 0 else None

        fig_path, ax_path = plt.subplots(figsize=(8, 4))
        
        colors_red = ['red', 'darkred', 'tomato', 'orange']
        colors_blue = ['blue', 'darkblue', 'royalblue', 'cyan']
        
        multi_freq_data = {}
        single_freq_cum_data = {}
        
        def calc_smoothed_path_length(xs, ys):
            if len(xs) < 2:
                return 0.0
            if len(xs) == 2:
                return np.sum(np.hypot(np.diff(xs), np.diff(ys)))
            from scipy.interpolate import UnivariateSpline

            t = np.arange(len(xs), dtype=float)
            spline_order = min(3, len(xs) - 1)
            smooth_factor = max(1.0, 0.25 * len(xs))
            spline_x = UnivariateSpline(t, xs, k=spline_order, s=smooth_factor)
            spline_y = UnivariateSpline(t, ys, k=spline_order, s=smooth_factor)
            t_dense = np.linspace(0, len(xs) - 1, max(100, len(xs) * 20))
            xs_fit = spline_x(t_dense)
            ys_fit = spline_y(t_dense)
            return np.sum(np.hypot(np.diff(xs_fit), np.diff(ys_fit)))

        def calc_smoothed_path_points(xs, ys):
            if len(xs) < 2:
                return np.array(xs), np.array(ys)
            from scipy.interpolate import UnivariateSpline

            t = np.arange(len(xs), dtype=float)
            spline_order = min(3, len(xs) - 1)
            smooth_factor = max(1.0, 0.25 * len(xs))
            spline_x = UnivariateSpline(t, xs, k=spline_order, s=smooth_factor)
            spline_y = UnivariateSpline(t, ys, k=spline_order, s=smooth_factor)
            t_dense = np.linspace(0, len(xs) - 1, max(100, len(xs) * 20))
            xs_fit = spline_x(t_dense)
            ys_fit = spline_y(t_dense)
            return xs_fit, ys_fit
        
        def calc_path_length_std_dev(per_particle_centers_list, noact_centers_list, use_fit=False):
            """Calculate standard deviation of path lengths across particles."""
            path_lengths = []
            for particle_centers in per_particle_centers_list:
                if particle_centers.shape[0] > 0:
                    cx = particle_centers[:, 0]
                    cy = particle_centers[:, 1]
                    # Get corresponding noact center for this particle
                    if len(noact_centers_list) > 0:
                        # Use the first noact center or average
                        no_x = np.mean([nc[0] for nc in noact_centers_list if nc.shape[0] > 0])
                        no_y = np.mean([nc[1] for nc in noact_centers_list if nc.shape[0] > 0])
                    else:
                        no_x, no_y = cx[0], cy[0]
                    
                    px = [no_x] + list(cx)
                    py = [no_y] + list(cy)
                    if use_fit and len(px) > 2:
                        px_fit, py_fit = calc_smoothed_path_points(px, py)
                        path_len = np.sum(np.hypot(np.diff(px_fit), np.diff(py_fit)))
                    else:
                        path_len = np.sum(np.hypot(np.diff(px), np.diff(py)))
                    path_lengths.append(path_len * (1/0.1928))
            
            if len(path_lengths) > 1:
                return np.std(path_lengths)
            return 0.0

        for i in range(len(freq_amp_vals)):
            if len(center_r_split[i]) > 0:
                cx_r = center_r_split[i][:, 0]
                cy_r = center_r_split[i][:, 1]
                cx_b = center_b_split[i][:, 0]
                cy_b = center_b_split[i][:, 1]
                no_r_x = noact_center_r_split[i][:, 0]
                no_r_y = noact_center_r_split[i][:, 1]
                no_b_x = noact_center_b_split[i][:, 0]
                no_b_y = noact_center_b_split[i][:, 1]
                
                amps = freq_amp_vals[i]
                path_len_r = []
                path_len_b = []
                path_len_r_fit = []
                path_len_b_fit = []
                
                for k in range(len(amps)):
                    px_r = [no_r_x[k]] + list(cx_r[0:k+1])
                    py_r = [no_r_y[k]] + list(cy_r[0:k+1])
                    raw_r = np.sum(np.hypot(np.diff(px_r), np.diff(py_r)))
                    path_len_r.append(raw_r * (1/0.1928))
                    if use_path_fit:
                        fit_r = calc_smoothed_path_length(px_r, py_r)
                        path_len_r_fit.append(fit_r * (1/0.1928))
                    else:
                        path_len_r_fit.append(raw_r * (1/0.1928))
                    
                    px_b = [no_b_x[k]] + list(cx_b[0:k+1])
                    py_b = [no_b_y[k]] + list(cy_b[0:k+1])
                    raw_b = np.sum(np.hypot(np.diff(px_b), np.diff(py_b)))
                    path_len_b.append(raw_b * (1/0.1928))
                    if use_path_fit:
                        fit_b = calc_smoothed_path_length(px_b, py_b)
                        path_len_b_fit.append(fit_b * (1/0.1928))
                    else:
                        path_len_b_fit.append(raw_b * (1/0.1928))
                    
                current_amps = np.array(amps, dtype=float)
                idt_num = int(idt_freqs[i][0])
                if idt_num == 1:
                    current_amps = current_amps / fd_idt1_scale
                elif idt_num == 2:
                    current_amps = current_amps / fd_idt2_scale
                elif idt_num == 3:
                    current_amps = current_amps / fd_idt3_scale
                elif idt_num == 4:
                    current_amps = current_amps / fd_idt4_scale
                
                if use_voltage:
                    current_amps = current_amps * conversion_rate
                    
                if len(frequency_values) > 1:
                    path_len_avg = np.mean([path_len_r, path_len_b], axis=0)
                    c_avg = colors_red[i % len(colors_red)]
                    idt_label = f'Avg IDT {int(idt_freqs[i][0])}'
                    
                    # Calculate std dev if enabled
                    yerr = None
                    if show_std_dev and center_r_per_particle_split is not None and center_b_per_particle_split is not None and i < len(center_r_per_particle_split):
                        # Calculate path length std dev from per-particle data
                        path_std_deviations = []
                        
                        # Get lists of per-particle arrays for this frequency
                        r_particles_by_amp = center_r_per_particle_split[i]  # List of arrays, one per amplitude
                        b_particles_by_amp = center_b_per_particle_split[i]
                        noact_r_array = noact_center_r_per_particle_split[i] if noact_center_r_per_particle_split is not None and i < len(noact_center_r_per_particle_split) else None
                        noact_b_array = noact_center_b_per_particle_split[i] if noact_center_b_per_particle_split is not None and i < len(noact_center_b_per_particle_split) else None
                        
                        if noact_r_array is not None and noact_b_array is not None and len(r_particles_by_amp) > 0:
                            for k in range(len(amps)):
                                # Get particle arrays at this amplitude step - shape (num_particles, 2)
                                r_pos_array = r_particles_by_amp[k] if k < len(r_particles_by_amp) else r_particles_by_amp[-1]
                                b_pos_array = b_particles_by_amp[k] if k < len(b_particles_by_amp) else b_particles_by_amp[-1]
                                
                                # Get noact positions (should be same for all amps) - shape (num_particles, 2)
                                noact_r_pos = noact_r_array[0] if len(noact_r_array) > 0 else None
                                noact_b_pos = noact_b_array[0] if len(noact_b_array) > 0 else None
                                
                                if noact_r_pos is not None and noact_b_pos is not None:
                                    # Calculate path lengths for each particle
                                    particle_paths = []
                                    # Use minimum number of particles that exist in all arrays
                                    num_particles = min(len(r_pos_array), len(noact_r_pos), len(b_pos_array), len(noact_b_pos))
                                    
                                    for p_idx in range(num_particles):
                                        # Red channel: collect positions from step 0 to k
                                        px_r = [noact_r_pos[p_idx, 0]] + [r_particles_by_amp[j][p_idx, 0] for j in range(min(k+1, len(r_particles_by_amp))) if p_idx < len(r_particles_by_amp[j])]
                                        py_r = [noact_r_pos[p_idx, 1]] + [r_particles_by_amp[j][p_idx, 1] for j in range(min(k+1, len(r_particles_by_amp))) if p_idx < len(r_particles_by_amp[j])]
                                        path_r = np.sum(np.hypot(np.diff(px_r), np.diff(py_r))) * (1/0.1928) if len(px_r) > 1 else 0
                                        
                                        # Blue channel
                                        px_b = [noact_b_pos[p_idx, 0]] + [b_particles_by_amp[j][p_idx, 0] for j in range(min(k+1, len(b_particles_by_amp))) if p_idx < len(b_particles_by_amp[j])]
                                        py_b = [noact_b_pos[p_idx, 1]] + [b_particles_by_amp[j][p_idx, 1] for j in range(min(k+1, len(b_particles_by_amp))) if p_idx < len(b_particles_by_amp[j])]
                                        path_b = np.sum(np.hypot(np.diff(px_b), np.diff(py_b))) * (1/0.1928) if len(px_b) > 1 else 0
                                        
                                        # Average red and blue
                                        path_avg = (path_r + path_b) / 2.0
                                        particle_paths.append(path_avg)
                                    
                                    if len(particle_paths) > 1:
                                        path_std_deviations.append(np.std(particle_paths))
                                    else:
                                        path_std_deviations.append(0.0)
                        
                        if len(path_std_deviations) == len(path_len_avg):
                            yerr = path_std_deviations
                    
                    # Plot with or without error bars
                    if yerr is not None:
                        ax_path.errorbar(current_amps, path_len_avg, yerr=yerr, fmt='o-', color=c_avg, label=idt_label, capsize=5, alpha=0.7)
                    else:
                        ax_path.plot(current_amps, path_len_avg, 'o-', color=c_avg, label=idt_label)
                    
                    if use_path_fit:
                        path_len_avg_fit = np.mean([path_len_r_fit, path_len_b_fit], axis=0)
                        ax_path.plot(current_amps, path_len_avg_fit, 'x--', color=c_avg, alpha=0.7, label=f'{idt_label} Fit')
                    
                    multi_freq_data[f"{idt_label} Amps"] = current_amps
                    multi_freq_data[f"{idt_label} Path (µm)"] = path_len_avg
                    if show_std_dev and yerr is not None:
                        multi_freq_data[f"{idt_label} Path Std Dev (µm)"] = np.array(yerr)
                    if use_path_fit:
                        multi_freq_data[f"{idt_label} Fit Path (µm)"] = path_len_avg_fit
                else:
                    # Single frequency case
                    yerr_r = None
                    yerr_b = None
                    
                    if show_std_dev and center_r_per_particle_split is not None and center_b_per_particle_split is not None and i < len(center_r_per_particle_split):
                        # Calculate path std dev for red and blue separately
                        path_std_r = []
                        path_std_b = []
                        
                        r_particles_by_amp = center_r_per_particle_split[i]
                        b_particles_by_amp = center_b_per_particle_split[i]
                        noact_r_array = noact_center_r_per_particle_split[i] if noact_center_r_per_particle_split is not None and i < len(noact_center_r_per_particle_split) else None
                        noact_b_array = noact_center_b_per_particle_split[i] if noact_center_b_per_particle_split is not None and i < len(noact_center_b_per_particle_split) else None
                        
                        if noact_r_array is not None and noact_b_array is not None and len(r_particles_by_amp) > 0:
                            for k in range(len(amps)):
                                r_particles = r_particles_by_amp[k] if k < len(r_particles_by_amp) else r_particles_by_amp[-1]
                                b_particles = b_particles_by_amp[k] if k < len(b_particles_by_amp) else b_particles_by_amp[-1]
                                noact_r = noact_r_array[0] if len(noact_r_array) > 0 else None
                                noact_b = noact_b_array[0] if len(noact_b_array) > 0 else None
                                
                                if noact_r is not None and noact_b is not None:
                                    # Use minimum number of particles that exist in all arrays
                                    num_particles = min(len(r_particles), len(noact_r), len(b_particles), len(noact_b))
                                    # Red channel
                                    r_paths = []
                                    for p_idx in range(num_particles):
                                        px = [noact_r[p_idx, 0]] + [r_particles_by_amp[j][p_idx, 0] for j in range(min(k+1, len(r_particles_by_amp))) if p_idx < len(r_particles_by_amp[j])]
                                        py = [noact_r[p_idx, 1]] + [r_particles_by_amp[j][p_idx, 1] for j in range(min(k+1, len(r_particles_by_amp))) if p_idx < len(r_particles_by_amp[j])]
                                        r_paths.append(np.sum(np.hypot(np.diff(px), np.diff(py))) * (1/0.1928) if len(px) > 1 else 0)
                                    
                                    # Blue channel
                                    b_paths = []
                                    for p_idx in range(num_particles):
                                        px = [noact_b[p_idx, 0]] + [b_particles_by_amp[j][p_idx, 0] for j in range(min(k+1, len(b_particles_by_amp))) if p_idx < len(b_particles_by_amp[j])]
                                        py = [noact_b[p_idx, 1]] + [b_particles_by_amp[j][p_idx, 1] for j in range(min(k+1, len(b_particles_by_amp))) if p_idx < len(b_particles_by_amp[j])]
                                        b_paths.append(np.sum(np.hypot(np.diff(px), np.diff(py))) * (1/0.1928) if len(px) > 1 else 0)
                                    
                                    path_std_r.append(np.std(r_paths) if len(r_paths) > 1 else 0.0)
                                    path_std_b.append(np.std(b_paths) if len(b_paths) > 1 else 0.0)
                        
                        if len(path_std_r) == len(path_len_r):
                            yerr_r = path_std_r
                        if len(path_std_b) == len(path_len_b):
                            yerr_b = path_std_b
                    
                    # Plot with error bars if available
                    if yerr_r is not None:
                        ax_path.errorbar(current_amps, path_len_r, yerr=yerr_r, fmt='o-', color='red', label=f'Red IDT {int(idt_freqs[i][0])}', capsize=5, alpha=0.7)
                    else:
                        ax_path.plot(current_amps, path_len_r, 'o-', color='red', label=f'Red IDT {int(idt_freqs[i][0])}')
                    
                    if yerr_b is not None:
                        ax_path.errorbar(current_amps, path_len_b, yerr=yerr_b, fmt='o--', color='blue', label=f'Blue IDT {int(idt_freqs[i][0])}', capsize=5, alpha=0.7)
                    else:
                        ax_path.plot(current_amps, path_len_b, 'o--', color='blue', label=f'Blue IDT {int(idt_freqs[i][0])}')
                    
                    if use_path_fit:
                        ax_path.plot(current_amps, path_len_r_fit, 'x-', color='darkred', alpha=0.7, label=f'Red Fit {int(idt_freqs[i][0])}')
                        ax_path.plot(current_amps, path_len_b_fit, 'x--', color='darkblue', alpha=0.7, label=f'Blue Fit {int(idt_freqs[i][0])}')
                    
                    single_freq_cum_data[f"IDT {int(idt_freqs[i][0])} Amps"] = current_amps
                    single_freq_cum_data[f"IDT {int(idt_freqs[i][0])} Red Path (µm)"] = path_len_r
                    single_freq_cum_data[f"IDT {int(idt_freqs[i][0])} Blue Path (µm)"] = path_len_b
                    if show_std_dev:
                        if yerr_r is not None:
                            single_freq_cum_data[f"IDT {int(idt_freqs[i][0])} Red Path Std Dev (µm)"] = np.array(yerr_r)
                        if yerr_b is not None:
                            single_freq_cum_data[f"IDT {int(idt_freqs[i][0])} Blue Path Std Dev (µm)"] = np.array(yerr_b)
                    if use_path_fit:
                        single_freq_cum_data[f"IDT {int(idt_freqs[i][0])} Red Fit Path (µm)"] = path_len_r_fit
                        single_freq_cum_data[f"IDT {int(idt_freqs[i][0])} Blue Fit Path (µm)"] = path_len_b_fit
        
        if use_voltage:
            ax_path.set_xlabel(r"Amplified voltage, $V_{out}$, V")
        else:
            ax_path.set_xlabel("Acoustic Amplitude")
        ax_path.set_ylabel("Cumulative Path Length (µm)")
        ax_path.tick_params(direction='out')
        ax_path.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_path.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_path)
        
        if len(frequency_values) > 1 and len(multi_freq_data) > 0:
            df_export = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in multi_freq_data.items() ]))
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Plot Data as CSV",
                data=csv,
                file_name='multi_freq_fluid_displacement_data.csv',
                mime='text/csv'
            )
        elif len(frequency_values) == 1 and len(single_freq_cum_data) > 0:
            df_export = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in single_freq_cum_data.items() ]))
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cumulative Plot Data as CSV",
                data=csv,
                file_name='single_freq_cumulative_data.csv',
                mime='text/csv'
            )

        st.subheader("Relative CoM Shift per Actuation Step")
        st.caption(
            "Absolute displacement magnitude of the centre of mass "
            "**relative to the previous actuation** (√(ΔX²+ΔZ²), µm). "
            "Step 0 is measured from the NoACT baseline."
        )

        rel_shift_data = {}
        xaxis_label = r"Amplified voltage, $V_{out}$, V" if use_voltage else "Acoustic amplitude"
        fig_rel, ax_rel = plt.subplots(figsize=(8, 4))
        has_rel_data = False

        for i in range(len(freq_amp_vals)):
            if len(center_r_split[i]) == 0:
                continue

            current_amps = np.array(freq_amp_vals[i], dtype=float)
            idt_num = int(idt_freqs[i][0])
            if idt_num == 1:
                current_amps = current_amps / fd_idt1_scale
            elif idt_num == 2:
                current_amps = current_amps / fd_idt2_scale
            elif idt_num == 3:
                current_amps = current_amps / fd_idt3_scale
            elif idt_num == 4:
                current_amps = current_amps / fd_idt4_scale
            if use_voltage:
                current_amps = current_amps * conversion_rate

            n_steps = len(current_amps)
            scale = 1 / 0.1928  # px -> µm
            rel_r = np.zeros((n_steps, 2))
            rel_b = np.zeros((n_steps, 2))

            # Step 0: relative to NoACT baseline
            rel_r[0] = (center_r_split[i][0] - noact_center_r_split[i][0]) * scale
            rel_b[0] = (center_b_split[i][0] - noact_center_b_split[i][0]) * scale

            # Steps k > 0: relative to the previous actuation step
            for k in range(1, n_steps):
                rel_r[k] = (center_r_split[i][k] - center_r_split[i][k - 1]) * scale
                rel_b[k] = (center_b_split[i][k] - center_b_split[i][k - 1]) * scale

            # Absolute magnitude averaged across channels
            mag_r = np.hypot(rel_r[:, 0], rel_r[:, 1])
            mag_b = np.hypot(rel_b[:, 0], rel_b[:, 1])
            mag_avg = (mag_r + mag_b) / 2.0

            idt_key = f"IDT {int(idt_freqs[i][0])}"
            rel_shift_data[f"{idt_key} Amps"] = current_amps
            rel_shift_data[f"{idt_key} Avg |Δr| (µm)"] = mag_avg
            
            # Calculate std dev for relative shift if show_std_dev is enabled
            yerr_rel = None
            if show_std_dev and center_r_per_particle_split is not None and center_b_per_particle_split is not None and i < len(center_r_per_particle_split):
                rel_stds = []
                r_particles_by_amp = center_r_per_particle_split[i]
                b_particles_by_amp = center_b_per_particle_split[i]
                noact_r_array = noact_center_r_per_particle_split[i] if noact_center_r_per_particle_split is not None and i < len(noact_center_r_per_particle_split) else None
                noact_b_array = noact_center_b_per_particle_split[i] if noact_center_b_per_particle_split is not None and i < len(noact_center_b_per_particle_split) else None
                
                if noact_r_array is not None and noact_b_array is not None and len(r_particles_by_amp) > 0:
                    for k in range(n_steps):
                        r_particles = r_particles_by_amp[k] if k < len(r_particles_by_amp) else r_particles_by_amp[-1]
                        b_particles = b_particles_by_amp[k] if k < len(b_particles_by_amp) else b_particles_by_amp[-1]
                        
                        if k == 0:
                            # First step: relative to NoACT
                            noact_r = noact_r_array[0] if len(noact_r_array) > 0 else None
                            noact_b = noact_b_array[0] if len(noact_b_array) > 0 else None
                            
                            if noact_r is not None and noact_b is not None:
                                # Use minimum number of particles that exist in all arrays
                                num_particles = min(len(r_particles), len(noact_r), len(b_particles), len(noact_b))
                                r_mags = []
                                for p_idx in range(num_particles):
                                    dr = (r_particles[p_idx, 0] - noact_r[p_idx, 0])
                                    dz = (r_particles[p_idx, 1] - noact_r[p_idx, 1])
                                    r_mags.append(np.hypot(dr, dz) * (1/0.1928))
                                
                                b_mags = []
                                for p_idx in range(num_particles):
                                    dr = (b_particles[p_idx, 0] - noact_b[p_idx, 0])
                                    dz = (b_particles[p_idx, 1] - noact_b[p_idx, 1])
                                    b_mags.append(np.hypot(dr, dz) * (1/0.1928))
                                
                                all_mags = r_mags + b_mags
                                rel_stds.append(np.std(all_mags) if len(all_mags) > 1 else 0.0)
                        else:
                            # Subsequent steps: relative to previous step
                            prev_r = r_particles_by_amp[k-1] if k-1 >= 0 and k-1 < len(r_particles_by_amp) else r_particles_by_amp[-1]
                            prev_b = b_particles_by_amp[k-1] if k-1 >= 0 and k-1 < len(b_particles_by_amp) else b_particles_by_amp[-1]
                            
                            # Use minimum number of particles that exist in all arrays
                            num_particles = min(len(r_particles), len(prev_r), len(b_particles), len(prev_b))
                            r_mags = []
                            for p_idx in range(num_particles):
                                dr = (r_particles[p_idx, 0] - prev_r[p_idx, 0])
                                dz = (r_particles[p_idx, 1] - prev_r[p_idx, 1])
                                r_mags.append(np.hypot(dr, dz) * (1/0.1928))
                            
                            b_mags = []
                            for p_idx in range(num_particles):
                                dr = (b_particles[p_idx, 0] - prev_b[p_idx, 0])
                                dz = (b_particles[p_idx, 1] - prev_b[p_idx, 1])
                                b_mags.append(np.hypot(dr, dz) * (1/0.1928))
                            
                            all_mags = r_mags + b_mags
                            rel_stds.append(np.std(all_mags) if len(all_mags) > 1 else 0.0)
                
                if len(rel_stds) == len(mag_avg):
                    yerr_rel = rel_stds
                    rel_shift_data[f"{idt_key} Std Dev |Δr| (µm)"] = np.array(rel_stds)
            
            if yerr_rel is not None:
                ax_rel.errorbar(current_amps, mag_avg, yerr=yerr_rel, fmt='o-', label=idt_key, capsize=5, alpha=0.7)
            else:
                ax_rel.plot(current_amps, mag_avg, 'o-', label=idt_key)
            has_rel_data = True

        if has_rel_data:
            ax_rel.set_ylabel("|Δr| (µm)")
            ax_rel.set_xlabel(xaxis_label)
            ax_rel.set_title("Absolute CoM Shift per Actuation Step")
            ax_rel.set_ylim(bottom=0)
            ax_rel.tick_params(direction='out')
            ax_rel.grid(True, linestyle='--', alpha=0.4)
            ax_rel.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_rel)

            df_rel = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rel_shift_data.items()]))
            csv_rel = df_rel.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Relative Shift Data as CSV",
                data=csv_rel,
                file_name='relative_shift_per_step.csv',
                mime='text/csv'
            )
        else:
            plt.close(fig_rel)

        if len(frequency_values) > 1:
            st.subheader("Red Channel Step Displacement (Multi-Frequency)")
            st.caption(
                "Step displacement is computed between consecutive actuation steps for the Red channel "
                "(Step 0 relative to NoACT baseline)."
            )

            fig_multi_xz, ax_multi_xz = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
            ax_x = ax_multi_xz[0, 0]
            ax_z = ax_multi_xz[0, 1]
            multi_freq_xz_data = {}
            has_multi_xz_data = False
            scale = 1 / 0.1928

            for i in range(len(freq_amp_vals)):
                if len(center_r_split[i]) == 0:
                    continue

                current_amps = np.array(freq_amp_vals[i], dtype=float)
                idt_num = int(idt_freqs[i][0])
                if idt_num == 1:
                    current_amps = current_amps / fd_idt1_scale
                elif idt_num == 2:
                    current_amps = current_amps / fd_idt2_scale
                elif idt_num == 3:
                    current_amps = current_amps / fd_idt3_scale
                elif idt_num == 4:
                    current_amps = current_amps / fd_idt4_scale
                if use_voltage:
                    current_amps = current_amps * conversion_rate

                n_steps = len(current_amps)
                step_dx = np.zeros(n_steps)
                step_dz = np.zeros(n_steps)
                step_dx_std = np.zeros(n_steps)
                step_dz_std = np.zeros(n_steps)

                # Mean values from averaged centers
                step_dx[0] = (center_r_split[i][0, 0] - noact_center_r_split[i][0, 0]) * scale
                step_dz[0] = (center_r_split[i][0, 1] - noact_center_r_split[i][0, 1]) * scale
                for k in range(1, n_steps):
                    step_dx[k] = (center_r_split[i][k, 0] - center_r_split[i][k - 1, 0]) * scale
                    step_dz[k] = (center_r_split[i][k, 1] - center_r_split[i][k - 1, 1]) * scale

                # Std values from per-particle data, when available
                if (
                    center_r_per_particle_split is not None
                    and noact_center_r_per_particle_split is not None
                    and i < len(center_r_per_particle_split)
                    and i < len(noact_center_r_per_particle_split)
                ):
                    r_particles_by_amp = center_r_per_particle_split[i]
                    noact_r_by_amp = noact_center_r_per_particle_split[i]
                    if len(r_particles_by_amp) > 0 and len(noact_r_by_amp) > 0:
                        for k in range(n_steps):
                            cur_r = r_particles_by_amp[k] if k < len(r_particles_by_amp) else r_particles_by_amp[-1]
                            if k == 0:
                                ref_r = noact_r_by_amp[0]
                            else:
                                ref_r = r_particles_by_amp[k - 1] if (k - 1) < len(r_particles_by_amp) else r_particles_by_amp[-1]

                            num_particles = min(len(cur_r), len(ref_r))
                            if num_particles > 0:
                                dx_vals = (cur_r[:num_particles, 0] - ref_r[:num_particles, 0]) * scale
                                dz_vals = (cur_r[:num_particles, 1] - ref_r[:num_particles, 1]) * scale
                                # Keep particle-based means in sync with std source when available.
                                step_dx[k] = float(np.mean(dx_vals))
                                step_dz[k] = float(np.mean(dz_vals))
                                step_dx_std[k] = float(np.std(dx_vals)) if num_particles > 1 else 0.0
                                step_dz_std[k] = float(np.std(dz_vals)) if num_particles > 1 else 0.0

                idt_key = f"IDT {int(idt_freqs[i][0])}"
                if show_std_dev:
                    ax_x.errorbar(current_amps, step_dx, yerr=step_dx_std, fmt='o-', capsize=4, alpha=0.8, label=idt_key)
                    ax_z.errorbar(current_amps, step_dz, yerr=step_dz_std, fmt='o-', capsize=4, alpha=0.8, label=idt_key)
                else:
                    ax_x.plot(current_amps, step_dx, 'o-', label=idt_key)
                    ax_z.plot(current_amps, step_dz, 'o-', label=idt_key)

                multi_freq_xz_data[f"{idt_key} Amps"] = current_amps
                multi_freq_xz_data[f"{idt_key} Red Step dX (um)"] = step_dx
                multi_freq_xz_data[f"{idt_key} Red Step dZ (um)"] = step_dz
                multi_freq_xz_data[f"{idt_key} Red Step dX Std (um)"] = step_dx_std
                multi_freq_xz_data[f"{idt_key} Red Step dZ Std (um)"] = step_dz_std
                has_multi_xz_data = True

            if has_multi_xz_data:
                xaxis_label = r"Amplified voltage, $V_{out}$, V" if use_voltage else "Acoustic amplitude"

                ax_x.set_xlabel(xaxis_label)
                ax_x.set_ylabel("Red Step dX (um)")
                ax_x.set_title("Red Channel Step X Displacement")
                ax_x.grid(True, linestyle='--', alpha=0.4)
                ax_x.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                ax_z.set_xlabel(xaxis_label)
                ax_z.set_ylabel("Red Step dZ (um)")
                ax_z.set_title("Red Channel Step Z Displacement")
                ax_z.grid(True, linestyle='--', alpha=0.4)
                ax_z.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.tight_layout()
                st.pyplot(fig_multi_xz)

                df_multi_xz = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in multi_freq_xz_data.items()]))
                csv_multi_xz = df_multi_xz.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Multi-Frequency Red Step X-Z Data as CSV",
                    data=csv_multi_xz,
                    file_name='multi_freq_red_step_xz_data.csv',
                    mime='text/csv'
                )
            else:
                plt.close(fig_multi_xz)

        if len(frequency_values) == 1:
            fig, ax = plt.subplots(len(freq_amp_vals), 2, figsize=(12, 4*len(freq_amp_vals)), squeeze=False)
            single_freq_xy_data = {}
            for i in range(len(freq_amp_vals)):
                if len(center_r_split[i]) > 0:
                    current_amps = np.array(freq_amp_vals[i], dtype=float)
                    if int(idt_freqs[i][0]) == 4:
                        current_amps = current_amps / fd_idt4_scale
                        
                    if use_voltage:
                        current_amps = current_amps * conversion_rate

                    scale = 1 / 0.1928
                    n_steps = len(current_amps)
                    step_dx = np.zeros(n_steps)
                    step_dz = np.zeros(n_steps)
                    step_bx = np.zeros(n_steps)
                    step_bz = np.zeros(n_steps)
                    step_dx_std = np.zeros(n_steps)
                    step_dz_std = np.zeros(n_steps)
                    step_bx_std = np.zeros(n_steps)
                    step_bz_std = np.zeros(n_steps)

                    # Mean step displacement from averaged centers
                    step_dx[0] = (center_r_split[i][0, 0] - noact_center_r_split[i][0, 0]) * scale
                    step_dz[0] = (center_r_split[i][0, 1] - noact_center_r_split[i][0, 1]) * scale
                    step_bx[0] = (center_b_split[i][0, 0] - noact_center_b_split[i][0, 0]) * scale
                    step_bz[0] = (center_b_split[i][0, 1] - noact_center_b_split[i][0, 1]) * scale
                    for k in range(1, n_steps):
                        step_dx[k] = (center_r_split[i][k, 0] - center_r_split[i][k - 1, 0]) * scale
                        step_dz[k] = (center_r_split[i][k, 1] - center_r_split[i][k - 1, 1]) * scale
                        step_bx[k] = (center_b_split[i][k, 0] - center_b_split[i][k - 1, 0]) * scale
                        step_bz[k] = (center_b_split[i][k, 1] - center_b_split[i][k - 1, 1]) * scale

                    # Std dev from per-particle data
                    if (
                        show_std_dev
                        and center_r_per_particle_split is not None
                        and center_b_per_particle_split is not None
                        and noact_center_r_per_particle_split is not None
                        and noact_center_b_per_particle_split is not None
                        and i < len(center_r_per_particle_split)
                        and i < len(center_b_per_particle_split)
                        and i < len(noact_center_r_per_particle_split)
                        and i < len(noact_center_b_per_particle_split)
                    ):
                        r_particles_by_amp = center_r_per_particle_split[i]
                        b_particles_by_amp = center_b_per_particle_split[i]
                        noact_r_array = noact_center_r_per_particle_split[i]
                        noact_b_array = noact_center_b_per_particle_split[i]
                        if len(r_particles_by_amp) > 0 and len(b_particles_by_amp) > 0 and len(noact_r_array) > 0 and len(noact_b_array) > 0:
                            for k in range(n_steps):
                                cur_r = r_particles_by_amp[k] if k < len(r_particles_by_amp) else r_particles_by_amp[-1]
                                cur_b = b_particles_by_amp[k] if k < len(b_particles_by_amp) else b_particles_by_amp[-1]
                                ref_r = noact_r_array[0] if k == 0 else (r_particles_by_amp[k - 1] if (k - 1) < len(r_particles_by_amp) else r_particles_by_amp[-1])
                                ref_b = noact_b_array[0] if k == 0 else (b_particles_by_amp[k - 1] if (k - 1) < len(b_particles_by_amp) else b_particles_by_amp[-1])

                                num_particles = min(len(cur_r), len(ref_r), len(cur_b), len(ref_b))
                                if num_particles > 0:
                                    dx_vals = (cur_r[:num_particles, 0] - ref_r[:num_particles, 0]) * scale
                                    dz_vals = (cur_r[:num_particles, 1] - ref_r[:num_particles, 1]) * scale
                                    bx_vals = (cur_b[:num_particles, 0] - ref_b[:num_particles, 0]) * scale
                                    bz_vals = (cur_b[:num_particles, 1] - ref_b[:num_particles, 1]) * scale
                                    step_dx[k] = float(np.mean(dx_vals))
                                    step_dz[k] = float(np.mean(dz_vals))
                                    step_bx[k] = float(np.mean(bx_vals))
                                    step_bz[k] = float(np.mean(bz_vals))
                                    step_dx_std[k] = float(np.std(dx_vals)) if num_particles > 1 else 0.0
                                    step_dz_std[k] = float(np.std(dz_vals)) if num_particles > 1 else 0.0
                                    step_bx_std[k] = float(np.std(bx_vals)) if num_particles > 1 else 0.0
                                    step_bz_std[k] = float(np.std(bz_vals)) if num_particles > 1 else 0.0

                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Amps"] = current_amps
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Red Step dX (µm)"] = step_dx
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Red Step dZ (µm)"] = step_dz
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Blue Step dX (µm)"] = step_bx
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Blue Step dZ (µm)"] = step_bz
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Red Step dX Std Dev (µm)"] = step_dx_std
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Red Step dZ Std Dev (µm)"] = step_dz_std
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Blue Step dX Std Dev (µm)"] = step_bx_std
                    single_freq_xy_data[f"IDT {int(idt_freqs[i][0])} Blue Step dZ Std Dev (µm)"] = step_bz_std
                    
                    # Plot step displacement for both channels
                    xaxis_label = r"Amplified voltage, $V_{out}$, V" if use_voltage else "Acoustic amplitude"
                    
                    ax[i, 0].errorbar(current_amps, step_dx, yerr=step_dx_std if show_std_dev else None, fmt='o-', color='r', capsize=5, alpha=0.7 if show_std_dev else 1.0, label='Red Step dX')
                    ax[i, 0].errorbar(current_amps, step_bx, yerr=step_bx_std if show_std_dev else None, fmt='s--', color='b', capsize=5, alpha=0.7 if show_std_dev else 1.0, label='Blue Step dX')
                    ax[i, 0].set_ylabel("Step dX (µm)")
                    ax[i, 0].set_xlabel(xaxis_label)
                    ax[i, 0].set_title(f"IDT {int(idt_freqs[i][0])} - Step X Displacement")
                    ax[i, 0].legend()
                    
                    # Z Axis Plot
                    ax[i, 1].errorbar(current_amps, step_dz, yerr=step_dz_std if show_std_dev else None, fmt='o-', color='r', capsize=5, alpha=0.7 if show_std_dev else 1.0, label='Red Step dZ')
                    ax[i, 1].errorbar(current_amps, step_bz, yerr=step_bz_std if show_std_dev else None, fmt='s--', color='b', capsize=5, alpha=0.7 if show_std_dev else 1.0, label='Blue Step dZ')
                    ax[i, 1].set_ylabel("Step dZ (µm)")
                    ax[i, 1].set_xlabel(xaxis_label)
                    ax[i, 1].set_title(f"IDT {int(idt_freqs[i][0])} - Step Z Displacement")
                    ax[i, 1].legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            if len(single_freq_xy_data) > 0:
                df_xy = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in single_freq_xy_data.items() ]))
                csv_xy = df_xy.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Single-Frequency Step X-Z Data as CSV",
                    data=csv_xy,
                    file_name='single_freq_red_step_xz_data.csv',
                    mime='text/csv'
                )

            st.subheader("Red Flow Trajectory Overlays (Separate)")
            bg_split = np.split(all_bg_images, amp_diff_idx + 1)
            red_mask_split = np.split(all_act_red_masks, amp_diff_idx + 1)
            
            for i in range(len(freq_amp_vals)):
                if len(bg_split[i]) > 0:
                    amps = freq_amp_vals[i]
                    num_amps = len(amps)
                    
                    cols = min(6, num_amps)
                    rows = (num_amps + 5) // 6
                    
                    fig_img, axes = plt.subplots(rows, cols, figsize=(3.5 * cols + 1, 4.5 * rows), squeeze=False)
                    axes = axes.flatten()
                    
                    for ax in axes[num_amps:]:
                        ax.axis('off')
                    
                    cx_r = center_r_split[i][:, 0]
                    cy_r_real = frame_length - center_r_split[i][:, 1]
                    noact_x_r = noact_center_r_split[i][:, 0]
                    noact_y_r_real = frame_length - noact_center_r_split[i][:, 1]
                    
                    for k, amp in enumerate(amps):
                        ax = axes[k]
                        
                        mask = red_mask_split[i][k]
                        img_to_show = np.zeros((frame_length, frame_length, 3), dtype=np.uint8)
                        if np.any(mask):
                            img_to_show[mask > 0] = [255, 0, 0]
                        ax.imshow(img_to_show)
                        
                        path_x = [noact_x_r[k]] + list(cx_r[0:k+1])
                        path_y = [noact_y_r_real[k]] + list(cy_r_real[0:k+1])
                        if use_path_fit and len(path_x) > 1:
                            path_x_plot, path_y_plot = calc_smoothed_path_points(path_x, path_y)
                            title_suffix = " (fitted)"
                        else:
                            path_x_plot, path_y_plot = path_x, path_y
                            title_suffix = ""

                        ax.plot(path_x_plot, path_y_plot, color='yellow', linewidth=5, linestyle='--')
                        ax.scatter(path_x, path_y, color='white', s=150, alpha=0.95, edgecolor='none')

                        path_len_px = np.sum(np.hypot(np.diff(path_x_plot), np.diff(path_y_plot)))
                        path_len_um = path_len_px * (1/0.1928)

                        disp_px = np.hypot(path_x_plot[-1] - path_x_plot[0], path_y_plot[-1] - path_y_plot[0])
                        disp_um = disp_px * (1/0.1928)

                        ax.set_title(f"Acoustic Amp: {int(amp)}\nDist: {disp_um:.1f} µm | Path: {path_len_um:.1f} µm{title_suffix}", fontsize=10)
                        ax.axis('off')
                    
                    fig_img.suptitle(f"IDT {int(idt_freqs[i][0])} Component Breakdown", fontsize=16)
                    plt.tight_layout(h_pad=2.0)
                    st.pyplot(fig_img)
        else:
            st.subheader("Final Red Flow Trajectories (Multi-Frequency View)")
            bg_split = np.split(all_bg_images, amp_diff_idx + 1)
            red_mask_split = np.split(all_act_red_masks, amp_diff_idx + 1)
            
            cols = len(freq_amp_vals)
            fig_img, axes = plt.subplots(1, cols, figsize=(5 * cols, 5), squeeze=False)
            axes = axes.flatten()
            
            for i in range(len(freq_amp_vals)):
                if len(bg_split[i]) > 0:
                    amps = freq_amp_vals[i]
                    k = len(amps) - 1
                    ax = axes[i]
                    
                    mask = red_mask_split[i][k]
                    img_to_show = np.zeros((frame_length, frame_length, 3), dtype=np.uint8)
                    if np.any(mask):
                        img_to_show[mask > 0] = [255, 0, 0]
                    ax.imshow(img_to_show)
                    
                    cx_r = center_r_split[i][:, 0]
                    cy_r_real = frame_length - center_r_split[i][:, 1]
                    noact_x_r = noact_center_r_split[i][:, 0]
                    noact_y_r_real = frame_length - noact_center_r_split[i][:, 1]
                    
                    path_x = [noact_x_r[k]] + list(cx_r[0:k+1])
                    path_y = [noact_y_r_real[k]] + list(cy_r_real[0:k+1])
                    if use_path_fit and len(path_x) > 1:
                        path_x_plot, path_y_plot = calc_smoothed_path_points(path_x, path_y)
                        title_suffix = " (fitted)"
                    else:
                        path_x_plot, path_y_plot = path_x, path_y
                        title_suffix = ""

                    ax.plot(path_x_plot, path_y_plot, color='yellow', linewidth=5, linestyle='--')
                    ax.scatter(path_x, path_y, color='white', s=150, alpha=0.95, edgecolor='none')

                    path_len_px = np.sum(np.hypot(np.diff(path_x_plot), np.diff(path_y_plot)))
                    path_len_um = path_len_px * (1/0.1928)

                    disp_px = np.hypot(path_x_plot[-1] - path_x_plot[0], path_y_plot[-1] - path_y_plot[0])
                    disp_um = disp_px * (1/0.1928)

                    ax.set_title(f"IDT {int(idt_freqs[i][0])} (Final Amp: {int(amps[-1])})\nDist: {disp_um:.1f} µm | Path: {path_len_um:.1f} µm{title_suffix}", fontsize=10)
                    ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_img)

            import io as _io
            def fig_to_png_bytes(fig, facecolor='white'):
                buf = _io.BytesIO()
                fig.savefig(buf, format='png', dpi=500, bbox_inches='tight', facecolor=facecolor)
                buf.seek(0)
                return buf

            st.download_button(
                label="📥 Save Trajectory Images",
                data=fig_to_png_bytes(fig_img, facecolor='white'),
                file_name='trajectory_images_multi_frequency.png',
                mime='image/png'
            )

            if show_stacked_evolution:
                st.subheader("Stacked Flow Evolution Images (Multi-Frequency)")
                fig_stack, axes_stack = plt.subplots(1, cols, figsize=(5 * cols, 5), squeeze=False)
                fig_stack.patch.set_facecolor('black')
                axes_stack = axes_stack.flatten()
                for i in range(len(freq_amp_vals)):
                    ax = axes_stack[i]
                    ax.set_facecolor('black')
                    if len(bg_split[i]) == 0:
                        ax.axis('off')
                        continue
                    amps = freq_amp_vals[i]
                    sample_count = min(4, len(amps))
                    sample_indices = np.unique(np.linspace(0, len(amps) - 1, sample_count, dtype=int))
                    alpha_values = np.linspace(0.25, 0.65, len(sample_indices))
                    for j, sample_idx in enumerate(sample_indices):
                        mask = red_mask_split[i][sample_idx]
                        if np.any(mask):
                            rgba = np.zeros((frame_length, frame_length, 4), dtype=np.float32)
                            rgba[mask > 0] = [1.0, 0.0, 0.0, float(alpha_values[j])]
                            ax.imshow(rgba)
                    cx_r = center_r_split[i][:, 0]
                    cy_r_real = frame_length - center_r_split[i][:, 1]
                    noact_x_r = noact_center_r_split[i][:, 0]
                    noact_y_r_real = frame_length - noact_center_r_split[i][:, 1]
                    if len(amps) > 0:
                        k = len(amps) - 1
                        path_x = [noact_x_r[k]] + list(cx_r[0:k+1])
                        path_y = [noact_y_r_real[k]] + list(cy_r_real[0:k+1])
                        if use_path_fit and len(path_x) > 1:
                            path_x_plot, path_y_plot = calc_smoothed_path_points(path_x, path_y)
                            title_suffix = " (fitted)"
                        else:
                            path_x_plot, path_y_plot = path_x, path_y
                            title_suffix = ""
                        ax.plot(path_x_plot, path_y_plot, color='yellow', linewidth=5, linestyle='--')
                        ax.scatter(path_x, path_y, color='white', s=150, alpha=0.95, edgecolor='none')
                    ax.set_title(f"IDT {int(idt_freqs[i][0])} Evolution{title_suffix}", fontsize=10)
                    ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig_stack)
                st.download_button(
                    label="📥 Save Stacked Trajectory Images",
                    data=fig_to_png_bytes(fig_stack, facecolor='black'),
                    file_name='trajectory_images_stacked_multi_frequency.png',
                    mime='image/png'
                )

    elif analysis_type == 'moi':
        M_r_avg_split = np.split(M_r_avg, amp_diff_idx + 1)
        M_b_avg_split = np.split(M_b_avg, amp_diff_idx + 1)
        M_r_std_split = np.split(M_r_std, amp_diff_idx + 1)
        M_b_std_split = np.split(M_b_std, amp_diff_idx + 1)
        # Component index 2 is the polar second moment in the image plane.
        # For paper convention (cross-section xz, flow axis y), this is reported as I_y.
        M_r_noact_split = [M_r_avg_split[j][:, 2, 0] for j in range(len(freq_amp_vals))]
        M_b_noact_split = [M_b_avg_split[j][:, 2, 0] for j in range(len(freq_amp_vals))]
        moi_plot_data = {}

        import matplotlib as mpl
        original_rc_moi = mpl.rcParams.copy()
        mpl.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
        })
        cm_to_in = 1 / 2.54
        fig_moi, ax_moi = plt.subplots(1, 1, figsize=(7.5 * cm_to_in, 5 * cm_to_in))

        colors_r = ['#C0392B', '#922B21', '#E74C3C', '#D35400']
        colors_b = ['#2471A3', '#1A5276', '#5DADE2', '#1F618D']
        alphas   = [1.0, 0.8, 0.6, 0.4]

        # Collect all unique amps across IDTs for tick placement
        unique_amps_moi = sorted(set(
            float(a) for arr in freq_amp_vals for a in arr
        ))

        for j in range(len(freq_amp_vals)):
            current_amps = np.array(freq_amp_vals[j], dtype=float)
            if use_voltage:
                x_vals = current_amps * conversion_rate
            else:
                x_vals = current_amps

            alpha_j = alphas[j % len(alphas)]
            idt_label = f'IDT {int(idt_freqs[j][0])}'

            # NoACT value prepended at x=0
            noact_r_val = np.nanmean(M_r_noact_split[j])
            noact_b_val = np.nanmean(M_b_noact_split[j])

            x_r = np.concatenate([[0], x_vals])
            y_r = np.concatenate([[noact_r_val], M_r_avg_split[j][:, 2, 1]])
            e_r = np.concatenate([[0], M_r_std_split[j][:, 2, 1]])

            x_b = np.concatenate([[0], x_vals])
            y_b = np.concatenate([[noact_b_val], M_b_avg_split[j][:, 2, 1]])
            e_b = np.concatenate([[0], M_b_std_split[j][:, 2, 1]])

            moi_plot_data[f'{idt_label} X'] = x_r
            moi_plot_data[f'{idt_label} Red I_y (mm^4)'] = y_r
            moi_plot_data[f'{idt_label} Red I_y Std (mm^4)'] = e_r
            moi_plot_data[f'{idt_label} Blue I_y (mm^4)'] = y_b
            moi_plot_data[f'{idt_label} Blue I_y Std (mm^4)'] = e_b

            # ACT lines — Red and Blue channels (starting from NoACT at 0)
            ax_moi.errorbar(x_r, y_r, yerr=e_r,
                            fmt='o-', color=colors_r[j % len(colors_r)],
                            alpha=alpha_j, linewidth=1.2, markersize=3,
                            capsize=2, capthick=0.6, elinewidth=0.6,
                            label=f'{idt_label} Red')
            ax_moi.errorbar(x_b, y_b, yerr=e_b,
                            fmt='s--', color=colors_b[j % len(colors_b)],
                            alpha=alpha_j, linewidth=1.2, markersize=3,
                            capsize=2, capthick=0.6, elinewidth=0.6,
                            label=f'{idt_label} Blue')

        # X-axis ticks — always include 0
        if use_voltage:
            import math
            v_all = [a * conversion_rate for a in unique_amps_moi]
            v_max = max(v_all)
            step_v = 12.5
            tick_pos = list(np.arange(0, v_max + step_v * 0.5, step_v))
            ax_moi.set_xticks(tick_pos)
            ax_moi.set_xticklabels([f"{v:.4g}" for v in tick_pos])
            ax_moi.set_xlim(-step_v * 0.3, v_max + step_v * 0.3)
            ax_moi.set_xlabel(r"Amplified voltage, $V_{out}$, V")
        else:
            tick_pos = [0] + unique_amps_moi
            ax_moi.set_xticks(tick_pos)
            ax_moi.set_xticklabels([str(int(a)) for a in tick_pos])
            ax_moi.set_xlim(-5, max(unique_amps_moi) + 5)
            ax_moi.set_xlabel('Acoustic amplitude, a.u.')

        ax_moi.set_ylabel(r'$I_y\;(\mathrm{mm}^4)$')
        ax_moi.legend(frameon=False)
        ax_moi.tick_params(direction='out', length=3)
        plt.tight_layout()
        st.pyplot(fig_moi)

        if show_moi_channel_separated:
            st.subheader("Channel-Separated Three-Axis MoI Analysis")
            st.caption(
                "Rows correspond to I_x, I_z, and I_y (polar about flow axis). "
                "Columns are Red channel, Blue channel, and percent difference "
                "((Actuated - Non-Actuated) / Non-Actuated) * 100."
            )

            axis_indices = [0, 1, 2]
            axis_labels = [r'$I_x$', r'$I_z$', r'$I_y$']
            xaxis_label = r"Amplified voltage, $V_{out}$, V" if use_voltage else "Acoustic amplitude, a.u."

            fig_sep_width_in = 16 / 2.54
            fig_sep_height_in = fig_sep_width_in * (12 / 15)
            fig_sep, axes_sep = plt.subplots(3, 3, figsize=(fig_sep_width_in, fig_sep_height_in), squeeze=False)
            diff_plot_data = {}
            unique_amps_sep = sorted(set(float(a) for arr in freq_amp_vals for a in arr))

            for row_idx, (axis_idx, axis_label) in enumerate(zip(axis_indices, axis_labels)):
                ax_red = axes_sep[row_idx, 0]
                ax_blue = axes_sep[row_idx, 1]
                ax_diff = axes_sep[row_idx, 2]

                for j in range(len(freq_amp_vals)):
                    current_amps = np.array(freq_amp_vals[j], dtype=float)
                    x_vals = current_amps * conversion_rate if use_voltage else current_amps
                    alpha_j = alphas[j % len(alphas)]
                    idt_label = f'IDT {int(idt_freqs[j][0])}'

                    red_act = M_r_avg_split[j][:, axis_idx, 1]
                    red_noact = M_r_avg_split[j][:, axis_idx, 0]
                    blue_act = M_b_avg_split[j][:, axis_idx, 1]
                    blue_noact = M_b_avg_split[j][:, axis_idx, 0]
                    red_act_std = M_r_std_split[j][:, axis_idx, 1]
                    red_noact_std = M_r_std_split[j][:, axis_idx, 0]
                    blue_act_std = M_b_std_split[j][:, axis_idx, 1]
                    blue_noact_std = M_b_std_split[j][:, axis_idx, 0]

                    red_diff = red_act - red_noact
                    blue_diff = blue_act - blue_noact
                    # Percent difference relative to non-actuated value.
                    # Safe division to avoid warnings for zero noact values.
                    red_diff = np.where(np.abs(red_noact) > 1e-12, (red_diff / red_noact) * 100.0, np.nan)
                    blue_diff = np.where(np.abs(blue_noact) > 1e-12, (blue_diff / blue_noact) * 100.0, np.nan)

                    # Error propagation for f = 100 * ((A - N) / N) = 100 * (A/N - 1)
                    # sigma_f = 100 * sqrt((sigma_A/N)^2 + (A*sigma_N/N^2)^2)
                    red_diff_std = np.where(
                        np.abs(red_noact) > 1e-12,
                        100.0 * np.sqrt((red_act_std / red_noact) ** 2 + ((red_act * red_noact_std) / (red_noact ** 2)) ** 2),
                        np.nan,
                    )
                    blue_diff_std = np.where(
                        np.abs(blue_noact) > 1e-12,
                        100.0 * np.sqrt((blue_act_std / blue_noact) ** 2 + ((blue_act * blue_noact_std) / (blue_noact ** 2)) ** 2),
                        np.nan,
                    )

                    # Red channel: actuated vs non-actuated
                    ax_red.errorbar(
                        x_vals, red_act,
                        yerr=red_act_std,
                        fmt='o-', color=colors_r[j % len(colors_r)], alpha=alpha_j,
                        linewidth=1.2, markersize=2,
                        capsize=2, capthick=0.6, elinewidth=0.6,
                        label=f'{idt_label} Act'
                    )
                    ax_red.errorbar(
                        x_vals, red_noact,
                        yerr=red_noact_std,
                        fmt='o--', color='black', alpha=alpha_j,
                        linewidth=1.0, markersize=2,
                        capsize=2, capthick=0.6, elinewidth=0.6,
                        label=f'{idt_label} NoACT'
                    )

                    # Blue channel: actuated vs non-actuated
                    ax_blue.errorbar(
                        x_vals, blue_act,
                        yerr=blue_act_std,
                        fmt='s-', color=colors_b[j % len(colors_b)], alpha=alpha_j,
                        linewidth=1.2, markersize=2,
                        capsize=2, capthick=0.6, elinewidth=0.6,
                        label=f'{idt_label} Act'
                    )
                    ax_blue.errorbar(
                        x_vals, blue_noact,
                        yerr=blue_noact_std,
                        fmt='s--', color='black', alpha=alpha_j,
                        linewidth=1.0, markersize=2,
                        capsize=2, capthick=0.6, elinewidth=0.6,
                        label=f'{idt_label} NoACT'
                    )

                    # Difference plot: two lines (Red and Blue) of Act - NoACT
                    ax_diff.errorbar(
                        x_vals, red_diff,
                        yerr=red_diff_std,
                        fmt='o-', color=colors_r[j % len(colors_r)], alpha=alpha_j,
                        linewidth=1.2, markersize=2,
                        capsize=2, capthick=0.6, elinewidth=0.6,
                        label=f'{idt_label} Red Δ%'
                    )
                    ax_diff.errorbar(
                        x_vals, blue_diff,
                        yerr=blue_diff_std,
                        fmt='s--', color=colors_b[j % len(colors_b)], alpha=alpha_j,
                        linewidth=1.2, markersize=2,
                        capsize=2, capthick=0.6, elinewidth=0.6,
                        label=f'{idt_label} Blue Δ%'
                    )

                    diff_plot_data[f'{idt_label} X'] = x_vals
                    diff_plot_data[f'{idt_label} {axis_label} Red Act (mm^4)'] = red_act
                    diff_plot_data[f'{idt_label} {axis_label} Red Act Std (mm^4)'] = red_act_std
                    diff_plot_data[f'{idt_label} {axis_label} Red NoACT (mm^4)'] = red_noact
                    diff_plot_data[f'{idt_label} {axis_label} Red NoACT Std (mm^4)'] = red_noact_std
                    diff_plot_data[f'{idt_label} {axis_label} Blue Act (mm^4)'] = blue_act
                    diff_plot_data[f'{idt_label} {axis_label} Blue Act Std (mm^4)'] = blue_act_std
                    diff_plot_data[f'{idt_label} {axis_label} Blue NoACT (mm^4)'] = blue_noact
                    diff_plot_data[f'{idt_label} {axis_label} Blue NoACT Std (mm^4)'] = blue_noact_std
                    diff_plot_data[f'{idt_label} {axis_label} Red Δ% ((Act-NoACT)/NoACT*100)'] = red_diff
                    diff_plot_data[f'{idt_label} {axis_label} Red Δ% Std'] = red_diff_std
                    diff_plot_data[f'{idt_label} {axis_label} Blue Δ% ((Act-NoACT)/NoACT*100)'] = blue_diff
                    diff_plot_data[f'{idt_label} {axis_label} Blue Δ% Std'] = blue_diff_std

                ax_red.set_ylabel(rf'{axis_label} $(\mathrm{{mm}}^4)$')
                ax_blue.set_ylabel(rf'{axis_label} $(\mathrm{{mm}}^4)$')
                ax_diff.set_ylabel(rf'Δ{axis_label} (%)')

                ax_red.set_xlabel(xaxis_label)
                ax_blue.set_xlabel(xaxis_label)
                ax_diff.set_xlabel(xaxis_label)

                if use_voltage and len(unique_amps_sep) > 0:
                    x_max = max(unique_amps_sep) * conversion_rate
                    tick_pos = list(np.arange(0, x_max + 12.5 * 0.5, 12.5))
                    ax_red.set_xticks(tick_pos)
                    ax_blue.set_xticks(tick_pos)
                    ax_diff.set_xticks(tick_pos)
                    ax_red.set_xlim(-12.5 * 0.3, x_max + 12.5 * 0.3)
                    ax_blue.set_xlim(-12.5 * 0.3, x_max + 12.5 * 0.3)
                    ax_diff.set_xlim(-12.5 * 0.3, x_max + 12.5 * 0.3)

                ax_red.tick_params(direction='out', length=3)
                ax_blue.tick_params(direction='out', length=3)
                ax_diff.tick_params(direction='out', length=3)

                ax_red.legend(fontsize=5, frameon=False)
                ax_blue.legend(fontsize=5, frameon=False)
                ax_diff.legend(fontsize=5, frameon=False)

            plt.tight_layout()
            st.pyplot(fig_sep)

            import io as _io_sep
            legends_sep = []
            for ax_row in axes_sep:
                for ax_cur in ax_row:
                    leg = ax_cur.get_legend()
                    if leg is not None:
                        legends_sep.append(leg)
                        leg.remove()

            buf_sep = _io_sep.BytesIO()
            fig_sep.savefig(buf_sep, format='png', dpi=300, bbox_inches='tight')
            buf_sep.seek(0)

            st.download_button(
                label="Download Channel-Separated 3-Axis MoI Plot (300 DPI, No Legend)",
                data=buf_sep,
                file_name="moi_channel_separated_3axis_300dpi_nolegend.png",
                mime="image/png"
            )

            for leg in legends_sep:
                leg.set_visible(True)

            if len(diff_plot_data) > 0:
                df_sep_export = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in diff_plot_data.items()]))
                csv_sep = df_sep_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Channel-Separated 3-Axis MoI Data (CSV)",
                    data=csv_sep,
                    file_name="moi_channel_separated_3axis_data.csv",
                    mime="text/csv"
                )

        if len(moi_plot_data) > 0:
            df_moi_export = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in moi_plot_data.items()]))
            csv_moi = df_moi_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download MoI Plot Data (CSV)",
                data=csv_moi,
                file_name="moi_plot_data.csv",
                mime="text/csv"
            )

        # Clean download
        import io as _io
        leg = ax_moi.get_legend()
        if leg: leg.remove()
        ax_moi.set_xlabel('')
        ax_moi.set_ylabel('')
        buf_moi = _io.BytesIO()
        fig_moi.savefig(buf_moi, format='png', dpi=500, transparent=True, bbox_inches='tight')
        buf_moi.seek(0)
        st.download_button(
            label="Download Clean MoI Plot",
            data=buf_moi,
            file_name="moi_iy_clean.png",
            mime="image/png"
        )
        mpl.rcParams.update(original_rc_moi)
            
    st.success("Analysis Complete!")

tab1, tab2, tab3, tab4 = st.tabs(["Mixing Coefficient", "Uniformity", "Fluid Displacement", "Moment of Inertia"])

with tab1:
    st.header("Mixing Coefficient")
    st.markdown("""
    **Analysis Guide:**
    * **Single-Frequency Analysis:** Evaluates images and generates a single set of box plots specifically for that isolated IDT across your defined amplitudes.
    * **Multi-Frequency Analysis:** Simultaneously evaluates and generates color-coded, side-by-side grouped box plots for multiple IDTs per amplitude tick, seamlessly enabling visual cross-frequency comparison.
    * **Mixing Coefficient Interpretation:** The mixing coefficient is basically a measurement of the standard deviation across the channel. A value of **0.5 represents ideal separation** (completely unmixed fluids), whereas a value of **0.0 represents ideal mixing** (perfectly homogenous distribution).
    * **Plain Amplitude (a.u.):** The 8-bit generation of the initial sinusoidal signal, which acts as the input at the signal generator and is universally limited to 500 mV bounds.
    * **Amplified Voltage (V):** The pure signal mapped through a hardware amplifier gain of approximately 44 dB, resulting in a mathematical conversion ratio of approx. 0.624 V/units when device input level is chosen as 500 mV/units (e.g., 10 units equates $(500\ \mathrm{mV/unit})/127\ \mathrm{units}=39.3\ \mathrm{mV}$ and amplified to 6.22 V).
    """)
    tc1 = render_tab_header("mix", allow_separate_amps=False)

    col1, col2 = st.columns(2)
    with col1:
        use_voltage = st.checkbox("Show X-axis as Amplified Voltage")
    with col2:
        conversion_rate = st.number_input("Conversion Factor (V per unit)", value=0.624, step=0.001, min_value=0.0, format="%.3f") if use_voltage else 1.0

    if st.button("Run Mixing Analysis"):
        execute_analysis('mixing', tc1, use_voltage=use_voltage, conversion_rate=conversion_rate)

with tab2:
    st.header("Uniformity")
    st.markdown("""
    **Analysis Guide:**
    * **Plain Amplitude (a.u.):** The 8-bit generation of the initial sinusoidal signal, which acts as the input at the signal generator and is universally limited to 500 mV bounds.
    * **Amplified Voltage (V):** The pure signal mapped through a hardware amplifier gain of approximately 44 dB, resulting in a mathematical conversion ratio of approx. 0.624 V/units when device input level is chosen as 500 mV/units (e.g., 10 units equates $(500\ \mathrm{mV/unit})/127\ \mathrm{units}=39.3\ \mathrm{mV}$ and amplified to 6.22 V).
    """)
    tc2 = render_tab_header("uni", allow_separate_amps=False)
    
    colA, colB, colC = st.columns(3)
    with colA:
        use_voltage_uni = st.checkbox("Convert to Amplified Voltage", value=False, key="uni_volt")
    with colB:
        conversion_rate_uni = 1.0
        if use_voltage_uni:
            conversion_rate_uni = st.number_input("Conversion Factor (V per unit)", value=0.624, step=0.001, min_value=0.0, format="%.3f", key="uni_conv")
    with colC:
        strict_uni_calc = st.checkbox("Strict Uniformity (100% Flow Match)", value=False, key="strict_uni_calc", help="If enabled, stability requires identical geometric mapping across ALL valid frames (half=0). If disabled, requires matching in at least half the frames (half=1).")
    
    if st.button("Run Uniformity Analysis"):
        execute_analysis('uniformity', tc2, use_voltage=use_voltage_uni, conversion_rate=conversion_rate_uni, strict_uni_calc=strict_uni_calc)

with tab3:
    st.header("Fluid Displacement")
    st.markdown("""
    **Analysis Guide:**
    * **Single-Frequency Analysis:** Generates 1) the Cumulative Path Length plot (with separate Red/Blue channel paths), 2) individual X-Axis and Z-Axis displacement charts, and 3) detailed Red Flow Trajectory image overlays.
    * **Multi-Frequency Analysis:** Generates exactly one unified Cumulative Path Length plot comparing your selected frequencies. The Red and Blue path lengths are theoretically averaged into a single trace per frequency to provide a clean multi-IDT comparison. A unified **Relative CoM Shift per Actuation Step** plot is also shown with all selected IDTs on the same figure, and both plots support CSV download.
    * **Amplitudes & S11 Scaling:** Configure amplitude scaling factors for each IDT here. Scaling is required because different IDTs have different transduction efficiencies. **IDT 1** is from legacy substrate experiments and uses a baseline scale of 1.0. **IDT 2, 3, and 4** are newer designs with optimized transducers (scales: IDT 2 ≈ 0.69, IDT 3 ≈ 0.88, IDT 4 ≈ 1.65). These scaling factors are calculated from the reflection parameter (S11) measurements and normalize output across different IDT generations, enabling proper multi-frequency comparison.
    * **Plain Amplitude (a.u.):** The 8-bit generation of the initial sinusoidal signal, which acts as the input at the signal generator and is universally limited to 500 mV bounds.
    * **Amplified Voltage (V):** The pure signal mapped through a hardware amplifier gain of approximately 44 dB, resulting in a mathematical conversion ratio of approx. 0.624 V/units when device input level is chosen as 500 mV/units (e.g., 10 units equates $(500\ \mathrm{mV/unit})/127\ \mathrm{units}=39.3\ \mathrm{mV}$ and amplified to 6.22 V).
    """)
    tc3 = render_tab_header("fd", allow_separate_amps=True)
    
    # IDT Amplitude Scales (all 4 side by side)
    scale_col1, scale_col2, scale_col3, scale_col4 = st.columns(4)
    
    fd_idt1_scale = 1.0
    with scale_col1:
        if 1 in tc3['frequency_values']:
            fd_idt1_scale = st.number_input("IDT 1 Scale", value=1.0, min_value=0.1, step=0.05)
    
    fd_idt2_scale = 1.4
    with scale_col2:
        if 2 in tc3['frequency_values']:
            fd_idt2_scale = st.number_input("IDT 2 Scale", value=0.6875, min_value=0.1, step=0.05)
    
    fd_idt3_scale = 1.0
    with scale_col3:
        if 3 in tc3['frequency_values']:
            fd_idt3_scale = st.number_input("IDT 3 Scale", value=0.875, min_value=0.1, step=0.05)
    
    fd_idt4_scale = 1.65
    with scale_col4:
        if 4 in tc3['frequency_values']:
            fd_idt4_scale = st.number_input("IDT 4 Scale", value=1.65, min_value=0.1, step=0.05)
        
    col1, col2 = st.columns(2)
    with col1:
        use_voltage_fd = st.checkbox("Show X-axis as Amplified Voltage", key="fd_volt")
    with col2:
        conversion_rate_fd = st.number_input("Conversion Factor (V per unit)", value=0.624, step=0.001, min_value=0.0, format="%.3f", key="fd_cr") if use_voltage_fd else 1.0
        
    fd_use_own_center = st.checkbox("Use Image Moments Center", value=True, key="fd_center")
    fd_use_path_fit = st.checkbox("Use fitted curve for path calculation", value=False, key="fd_path_fit")
    fd_show_std_dev = st.checkbox("Show Standard Deviation (σ) Error Bars", value=False, key="fd_std_dev", help="Display standard deviation from multiple particle measurements.")
    fd_show_stacked_evolution = st.checkbox("Show stacked evolution images in multi-frequency view", value=False, key="fd_stacked")
    if st.button("Run Fluid Displacement"):
        execute_analysis(
            'fluid_dislocation', tc3,
            use_own_center=fd_use_own_center,
            fd_idt1_scale=fd_idt1_scale,
            fd_idt2_scale=fd_idt2_scale,
            fd_idt3_scale=fd_idt3_scale,
            fd_idt4_scale=fd_idt4_scale,
            use_voltage=use_voltage_fd,
            conversion_rate=conversion_rate_fd,
            use_path_fit=fd_use_path_fit,
            show_stacked_evolution=fd_show_stacked_evolution,
            show_std_dev=fd_show_std_dev
        )

with tab4:
    st.header("Moment of Inertia")
    st.markdown("""
    **Analysis Guide:**
    * **Axis Convention in This App (paper-aligned):** The plotted value is the polar second moment about the flow axis and is labeled as $I_y$ for an $x$-$z$ cross-section.
    * **Plain Amplitude (a.u.):** The 8-bit generation of the initial sinusoidal signal, which acts as the input at the signal generator and is universally limited to 500 mV bounds.
    * **Amplified Voltage (V):** The pure signal mapped through a hardware amplifier gain of approximately 44 dB, resulting in a mathematical conversion ratio of approx. 0.624 V/units when device input level is chosen as 500 mV/units (e.g., 10 units equates $(500\ \mathrm{mV/unit})/127\ \mathrm{units}=39.3\ \mathrm{mV}$ and amplified to 6.22 V).
    """)
    tc4 = render_tab_header("moi", allow_separate_amps=False, default_freq_mask="1000")
    moi_use_own_center = st.checkbox("Use Image Moments Center", value=True, key="moi_center")
    moi_show_channel_separated = st.checkbox(
        "Channel-separated three-axis MoI analysis (3x3)",
        value=False,
        key="moi_channel_sep"
    )
    colA_moi, colB_moi = st.columns(2)
    with colA_moi:
        use_voltage_moi = st.checkbox("Convert to Amplified Voltage", value=False, key="moi_volt")
    with colB_moi:
        conversion_rate_moi = 1.0
        if use_voltage_moi:
            conversion_rate_moi = st.number_input("Conversion Factor (V per unit)", value=0.624, step=0.001, min_value=0.0, format="%.3f", key="moi_conv")
    if st.button("Run Moment of Inertia"):
        execute_analysis('moi', tc4, use_own_center=moi_use_own_center,
                         use_voltage=use_voltage_moi, conversion_rate=conversion_rate_moi,
                         show_moi_channel_separated=moi_show_channel_separated)
