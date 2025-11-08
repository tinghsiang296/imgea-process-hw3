"""Streamlit interactive app for homography demo.

Usage:
  pip install -r requirements.txt
  streamlit run streamlit_app.py

This app uses streamlit-drawable-canvas to let the user click points on the image.
"""
import io
import base64
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import json
import os
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import traceback

from homography_utils import estimate_homography, warp_image
try:
    from streamlit_image_coords import image_coords
except Exception:
    image_coords = None


st.set_page_config(layout='wide')

# Set to True only when you want to collect and expose debug artifacts.
# Keep False for normal deployments to avoid leaving files/paths visible.
DEBUG_SAVE = False


def pil_from_bytes(data):
    return Image.open(io.BytesIO(data)).convert('RGB')


def pil_to_data_url(pil_image, fmt='PNG'):
    """Convert a PIL Image to a data URL (base64) for use as background_image.

    Using a data URL avoids relying on internal Streamlit APIs that some
    versions of streamlit-drawable-canvas call and which can break on deploy.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format=fmt)
    b = base64.b64encode(buffered.getvalue()).decode('ascii')
    return f"data:image/{fmt.lower()};base64,{b}"


def save_debug(info: dict, prefix: str = 'canvas_debug'):
    # Only persist debug artifacts when DEBUG_SAVE is enabled
    if not DEBUG_SAVE:
        return None
    try:
        debug_dir = os.path.join(os.getcwd(), 'canvas_debug')
        os.makedirs(debug_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        fname = os.path.join(debug_dir, f'{prefix}_{ts}.json')
        with open(fname, 'w', encoding='utf8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return fname
    except Exception as e:
        # best-effort: if saving JSON fails, try to save a text fallback
        try:
            fname = os.path.join(debug_dir, f'{prefix}_{ts}.txt')
            with open(fname, 'w', encoding='utf8') as f:
                f.write(repr(info))
            return fname
        except Exception:
            return None


def create_canvas_with_diagnostics(background, canvas_kwargs, prefix):
    """Try to create a canvas while saving diagnostic info if it fails.

    background: PIL Image or other object; canvas_kwargs: dict: kwargs for st_canvas
    Returns (canvas_result, debug_info_filename_or_None)
    """
    info = {'time': datetime.now().isoformat(), 'background_type': str(type(background)), 'background_repr': None, 'background_head': None, 'attempts': []}
    try:
        info['background_repr'] = repr(background)[:2000]
    except Exception:
        info['background_repr'] = '<repr-failed>'
    # try to capture a small head of bytes when possible
    try:
        if isinstance(background, Image.Image):
            info['pil_size'] = background.size
            info['pil_mode'] = background.mode
            # first few bytes of PNG/PNG header when saving to buffer
            buf = io.BytesIO()
            try:
                background.save(buf, format='PNG')
                b = buf.getvalue()[:256]
                info['background_head'] = base64.b64encode(b).decode('ascii')
            except Exception:
                info['background_head'] = '<save-failed>'
        elif isinstance(background, (bytes, bytearray)):
            info['background_head'] = base64.b64encode(background[:256]).decode('ascii')
        elif isinstance(background, str):
            info['background_head'] = background[:512]
    except Exception:
        info['background_head'] = '<head-capture-failed>'

    # First attempt: pass background as-is
    try:
        canvas = st_canvas(background_image=background, **canvas_kwargs)
        info['attempts'].append({'mode': 'pil-as-is', 'success': True})
        fname = save_debug(info, prefix + '_background_info')
        return canvas, fname
    except Exception as e:
        tb = traceback.format_exc()
        info['attempts'].append({'mode': 'pil-as-is', 'success': False, 'error': str(e), 'traceback': tb})

    # Second attempt: try data URL
    try:
        data_url = None
        try:
            if isinstance(background, Image.Image):
                data_url = pil_to_data_url(background.convert('RGBA'))
            elif isinstance(background, (bytes, bytearray)):
                # try to interpret as image bytes
                try:
                    pil = pil_from_bytes(background)
                    data_url = pil_to_data_url(pil.convert('RGBA'))
                except Exception:
                    data_url = None
            elif isinstance(background, str) and background.startswith('data:'):
                data_url = background
        except Exception as e:
            info['attempts'].append({'mode': 'to-data-url-failed', 'error': str(e)})

        if data_url is not None:
            try:
                canvas = st_canvas(background_image=data_url, **canvas_kwargs)
                info['attempts'].append({'mode': 'data-url', 'success': True})
                fname = save_debug(info, prefix + '_background_info')
                return canvas, fname
            except Exception as e:
                tb = traceback.format_exc()
                info['attempts'].append({'mode': 'data-url', 'success': False, 'error': str(e), 'traceback': tb})
    except Exception:
        pass

    # final: save info and return None
    fname = save_debug(info, prefix + '_background_info')
    return None, fname


st.title('Homography Interactive (Streamlit)')

# initialize session state lists for collected points (used for single-click)
if 'rect_points' not in st.session_state:
    st.session_state['rect_points'] = []
if 'proj_points' not in st.session_state:
    st.session_state['proj_points'] = []

mode = st.sidebar.radio('Mode', ['Rectify (拉正)', 'Project (貼圖投影)'], key='mode')
# let user choose canvas tool; freedraw is more reliable across browsers
# fixed UI: only freedraw canvas and Canvas input are supported per user request
tool = 'freedraw'
# input method forced to Canvas (no single-click alternatives)
input_mode = 'Canvas'

if mode.startswith('Rectify'):
    uploaded = st.sidebar.file_uploader('Upload image to rectify', type=['png', 'jpg', 'jpeg'])
    if uploaded is not None:
        # use getvalue() to safely obtain bytes and allow re-use
        canvas_result = None
        dbg_fname = None
        pil = pil_from_bytes(uploaded.getvalue())
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = pil.size[1], pil.size[0]

        # debug: show uploaded image and info
        st.image(pil, caption='Uploaded image (debug)')
        st.write('Image size (width, height):', pil.size)

        st.write('Click 4 points on the image (order: TL, TR, BR, BL or any consistent clockwise/counterclockwise)')
        st.write('Tool: choose "point" if your browser supports it; otherwise use "freedraw" and draw small dots.')
        use_single_click = st.sidebar.checkbox('Use single-click fallback (streamlit-image-coords)', key='use_click_rect')
        # limit canvas size to reasonable viewport to avoid rendering issues
        canvas_h = min(h, 900)
        canvas_w = min(w, 1200)
        # Pass a PIL image as background. Some streamlit-drawable-canvas
        # versions expect an image-like object with .height/.width.
        if not use_single_click:
            bg_pil = pil.convert('RGBA')
            canvas_kwargs = dict(
                fill_color='rgba(0,0,0,0)',
                stroke_width=3,
                stroke_color='#ff0000',
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode=tool,
                key='rect_canvas',
            )
            canvas_result, dbg_fname = create_canvas_with_diagnostics(bg_pil, canvas_kwargs, prefix='rect')
            if DEBUG_SAVE and dbg_fname:
                st.write(f'Canvas background diagnostic saved to {dbg_fname}')
        else:
            # show image and use single-click capture
            st.image(pil, caption='Uploaded image (click to select points)')
            if image_coords is None:
                st.warning('streamlit-image-coords not available. Install with: pip install streamlit-image-coords')
                canvas_result = None
            else:
                coords = None
                try:
                    try:
                        coords = image_coords(pil)
                    except TypeError:
                        coords = image_coords(image=pil)
                except Exception as e:
                    st.error(f'image_coords error: {e}')
                    coords = None
                if coords is not None:
                    st.write('image_coords result (debug):', coords)
                    if isinstance(coords, dict) and 'x' in coords and 'y' in coords:
                        cx, cy = float(coords['x']), float(coords['y'])
                    elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        cx, cy = float(coords[0]), float(coords[1])
                    else:
                        cx = cy = None
                    if cx is not None:
                        st.write(f'Current click: ({cx:.1f}, {cy:.1f})')
                        if st.button('Add point (click)', key='add_point_click_rect'):
                            st.session_state['rect_points'].append((cx, cy))
                st.write('Collected single-click points:', st.session_state['rect_points'])
                if st.button('Clear single-click points', key='clear_rect_points'):
                    st.session_state['rect_points'] = []
            if DEBUG_SAVE and 'dbg_fname' in locals() and dbg_fname:
                try:
                    with open(dbg_fname, 'r', encoding='utf8') as _f:
                        data = json.load(_f)
                    st.subheader('Canvas background diagnostic (preview)')
                    st.json(data)
                    try:
                        with open(dbg_fname, 'rb') as _f:
                            blob = _f.read()
                        st.download_button('Download background diagnostic JSON', data=blob, file_name=os.path.basename(dbg_fname), mime='application/json')
                    except Exception:
                        pass
                except Exception:
                    try:
                        with open(dbg_fname, 'r', encoding='utf8') as _f:
                            txt = _f.read()
                        st.text(txt[:10000])
                    except Exception:
                        pass

        # Additional runtime diagnostics about the returned canvas object
        try:
            if canvas_result is not None:
                cr_info = {'repr': repr(canvas_result)[:2000]}
                try:
                    cr_info['has_json'] = canvas_result.json_data is not None
                except Exception:
                    cr_info['has_json'] = None
                try:
                    cr_info['objects_len'] = len(canvas_result.json_data.get('objects', [])) if canvas_result.json_data else None
                except Exception:
                    cr_info['objects_len'] = None
                try:
                    has_img = hasattr(canvas_result, 'image_data') and canvas_result.image_data is not None
                    cr_info['has_image_data'] = bool(has_img)
                    if has_img:
                        try:
                            arr = canvas_result.image_data
                            cr_info['image_data_type'] = str(type(arr))
                            try:
                                cr_info['image_data_shape'] = getattr(arr, 'shape', None)
                            except Exception:
                                cr_info['image_data_shape'] = None
                        except Exception:
                            cr_info['image_data_shape'] = None
                except Exception:
                    cr_info['has_image_data'] = None
                fname2 = save_debug(cr_info, prefix='proj_canvas_result')
                if fname2:
                    st.write(f'Canvas runtime diagnostic saved to {fname2}')
                    try:
                        with open(fname2, 'rb') as _f:
                            blob2 = _f.read()
                        st.download_button('Download canvas runtime diagnostic JSON', data=blob2, file_name=os.path.basename(fname2), mime='application/json')
                    except Exception:
                        pass
        except Exception:
            pass

        # Additional runtime diagnostics about the returned canvas object
        try:
            if canvas_result is not None:
                cr_info = {'repr': repr(canvas_result)[:2000]}
                try:
                    cr_info['has_json'] = canvas_result.json_data is not None
                except Exception:
                    cr_info['has_json'] = None
                try:
                    cr_info['objects_len'] = len(canvas_result.json_data.get('objects', [])) if canvas_result.json_data else None
                except Exception:
                    cr_info['objects_len'] = None
                try:
                    has_img = hasattr(canvas_result, 'image_data') and canvas_result.image_data is not None
                    cr_info['has_image_data'] = bool(has_img)
                    if has_img:
                        try:
                            # try to capture shape/size without converting whole image
                            arr = canvas_result.image_data
                            cr_info['image_data_type'] = str(type(arr))
                            try:
                                cr_info['image_data_shape'] = getattr(arr, 'shape', None)
                            except Exception:
                                cr_info['image_data_shape'] = None
                        except Exception:
                            cr_info['image_data_shape'] = None
                except Exception:
                    cr_info['has_image_data'] = None
                fname2 = save_debug(cr_info, prefix='rect_canvas_result')
                if fname2:
                    st.write(f'Canvas runtime diagnostic saved to {fname2}')
        except Exception:
            pass

        # end single-click handling

        # If user used single-click fallback, prefer collected points from session state
        if use_single_click:
            points_found = list(st.session_state.get('rect_points', []))

        # debug: show raw canvas json and save to file for debugging
        if canvas_result is not None:
            try:
                debug_dir = os.path.join(os.getcwd(), 'canvas_debug')
                os.makedirs(debug_dir, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                fname = os.path.join(debug_dir, f'rect_canvas_{ts}.json')
                with open(fname, 'w', encoding='utf8') as f:
                    json.dump(canvas_result.json_data, f, ensure_ascii=False, indent=2)
                st.write(f'Canvas JSON saved to {fname}')
            except Exception as e:
                st.write(f'Could not save canvas json: {e}')

        # helper: try to extract point coordinates from a canvas object
        import re

        def obj_to_point(obj):
            # common simple case: objects from circle/ellipse/rect/freedraw have left/top
            if obj is None:
                return None
            if 'left' in obj and 'top' in obj:
                try:
                    return float(obj.get('left', 0)), float(obj.get('top', 0))
                except Exception:
                    pass
            # Freedraw/path objects often carry a 'path' string like 'M x y L x y ...'
            path = obj.get('path') if isinstance(obj, dict) else None
            if isinstance(path, str):
                nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", path)
                coords = [float(n) for n in nums]
                if len(coords) >= 2:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))
            # fallback: try bounding box
            if 'width' in obj and 'height' in obj and 'left' in obj and 'top' in obj:
                try:
                    cx = float(obj['left']) + float(obj['width']) / 2.0
                    cy = float(obj['top']) + float(obj['height']) / 2.0
                    return cx, cy
                except Exception:
                    pass
            return None

        points_found = points_found if 'points_found' in locals() else []
        if canvas_result is not None and getattr(canvas_result, 'json_data', None) is not None:
            objects = canvas_result.json_data.get('objects', [])
            for obj in objects:
                p = obj_to_point(obj)
                if p is not None:
                    points_found.append(p)

        if len(points_found) >= 4:
            pts = np.array(points_found[:4], dtype=np.float64)
            st.write('Detected points (from objects):')
            st.write(pts)
            if st.button('Run rectify'):
                # compute target size
                def dist(a, b):
                    return np.linalg.norm(a - b)
                w1 = dist(pts[0], pts[1])
                w2 = dist(pts[3], pts[2])
                h1 = dist(pts[0], pts[3])
                h2 = dist(pts[1], pts[2])
                W = int(round(max(w1, w2)))
                H = int(round(max(h1, h2)))
                dst_pts = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float64)
                Hmat = estimate_homography(pts, dst_pts)
                warped = warp_image(img_cv, Hmat, (H, W))
                st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption='Rectified')
        else:
            st.info('No points detected yet. If clicking does not create objects, try using the "freedraw" tool to draw small dots (draw small circles), then the app will extract their centroids as points. Alternatively enable the single-click fallback in the sidebar to pick points directly on the image.')

else:
    uploaded_tex = st.sidebar.file_uploader('Upload texture image', type=['png', 'jpg', 'jpeg'])
    uploaded_bg = st.sidebar.file_uploader('Upload background image', type=['png', 'jpg', 'jpeg'])
    if uploaded_tex is not None and uploaded_bg is not None:
        canvas_result = None
        dbg_fname = None
        tex = pil_from_bytes(uploaded_tex.getvalue())
        bg = pil_from_bytes(uploaded_bg.getvalue())
        bg_cv = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
        th, tw = tex.size[1], tex.size[0]
        bh, bw = bg.size[1], bg.size[0]

        # debug: show uploads
        st.image(tex, caption='Texture (debug)')
        st.image(bg, caption='Background (debug)')
        st.write('Texture size (w,h):', tex.size, 'Background size (w,h):', bg.size)

        st.write('Click 4 points on the background where the texture should map to.')
        st.write('Tool: choose "point" if your browser supports it; otherwise use "freedraw" and draw small dots.')
        # limit canvas size to reasonable viewport to avoid rendering issues
        canvas_h = min(bh, 900)
        canvas_w = min(bw, 1200)
        use_single_click_proj = st.sidebar.checkbox('Use single-click fallback (streamlit-image-coords) for Project', key='use_click_proj')
        if not use_single_click_proj:
            bg_pil = bg.convert('RGBA')
            canvas_kwargs = dict(
                fill_color='rgba(0,0,0,0)',
                stroke_width=3,
                stroke_color='#00ff00',
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode=tool,
                key='proj_canvas',
            )
            canvas_result, dbg_fname = create_canvas_with_diagnostics(bg_pil, canvas_kwargs, prefix='proj')
            if dbg_fname:
                st.write(f'Canvas background diagnostic saved to {dbg_fname}')
                try:
                    with open(dbg_fname, 'r', encoding='utf8') as _f:
                        data = json.load(_f)
                    st.subheader('Canvas background diagnostic (preview)')
                    st.json(data)
                except Exception:
                    try:
                        with open(dbg_fname, 'r', encoding='utf8') as _f:
                            txt = _f.read()
                        st.text(txt[:10000])
                    except Exception:
                        pass
        else:
            # Show background image and collect single-click points via image_coords
            st.image(bg, caption='Background (click to select points)')
            if image_coords is None:
                st.warning('streamlit-image-coords not available. Install with: pip install streamlit-image-coords')
                canvas_result = None
            else:
                coords = None
                try:
                    try:
                        coords = image_coords(bg)
                    except TypeError:
                        coords = image_coords(image=bg)
                except Exception as e:
                    st.error(f'image_coords error: {e}')
                    coords = None
                if coords is not None:
                    st.write('image_coords result (debug):', coords)
                    if isinstance(coords, dict) and 'x' in coords and 'y' in coords:
                        cx, cy = float(coords['x']), float(coords['y'])
                    elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        cx, cy = float(coords[0]), float(coords[1])
                    else:
                        cx = cy = None
                    if cx is not None:
                        st.write(f'Current click: ({cx:.1f}, {cy:.1f})')
                        if st.button('Add point (click)', key='add_point_click_proj'):
                            st.session_state['proj_points'].append((cx, cy))
                st.write('Collected single-click points (proj):', st.session_state['proj_points'])
                if st.button('Clear single-click points (proj)', key='clear_proj_points'):
                    st.session_state['proj_points'] = []

        # Note: single-click capture removed — only Canvas (freedraw) is supported

        # Save canvas JSON if available
        if canvas_result is not None:
            try:
                debug_dir = os.path.join(os.getcwd(), 'canvas_debug')
                os.makedirs(debug_dir, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                fname = os.path.join(debug_dir, f'proj_canvas_{ts}.json')
                with open(fname, 'w', encoding='utf8') as f:
                    json.dump(canvas_result.json_data, f, ensure_ascii=False, indent=2)
                st.write(f'Canvas JSON saved to {fname}')
            except Exception as e:
                st.write(f'Could not save canvas json: {e}')

        # Collect points either from single-click fallback or from canvas objects
        points_found = []
        if use_single_click_proj:
            points_found = list(st.session_state.get('proj_points', []))
        else:
            if canvas_result is not None and getattr(canvas_result, 'json_data', None) is not None:
                objects = canvas_result.json_data.get('objects', [])
                for obj in objects[:4]:
                    x = obj.get('left', 0)
                    y = obj.get('top', 0)
                    points_found.append([x, y])

        if len(points_found) >= 4:
            pts = np.array(points_found[:4], dtype=np.float64)
            st.write('Selected points:')
            st.write(pts)
            if st.button('Run project'):
                src_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float64)
                Hmat = estimate_homography(src_pts, pts)
                tex_cv = cv2.cvtColor(np.array(tex), cv2.COLOR_RGB2BGR)
                warped = warp_image(tex_cv, Hmat, (bg_cv.shape[0], bg_cv.shape[1]))
                mask = np.any(warped != 0, axis=2)
                comp = bg_cv.copy()
                comp[mask] = warped[mask]
                st.image(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB), caption='Projected')
