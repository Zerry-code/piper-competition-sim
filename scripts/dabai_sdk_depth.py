"""Read a depth frame using the locally downloaded official Orbbec SDK C API."""
import ctypes as c
import os
from pathlib import Path
import numpy as np

def capture_depth(output):
    root=Path(__file__).resolve().parents[1]
    choices=list((root/'tools/orbbec/sdk').rglob('OrbbecSDK.dll'))
    if not choices:return dict(ok=False,status='sdk_missing',message='未找到本地官方 OrbbecSDK.dll')
    dll=choices[0];directory=os.add_dll_directory(str(dll.parent));sdk=c.CDLL(str(dll))
    P=c.c_void_p;E=c.POINTER(P)
    def bind(name,result,*args):
        f=getattr(sdk,name);f.restype=result;f.argtypes=list(args)+[E];return f
    message=sdk.ob_error_message;message.restype=c.c_char_p;message.argtypes=[P]
    free_error=sdk.ob_delete_error;free_error.argtypes=[P]
    def call(name,result,args,values):
        error=P();value=bind(name,result,*args)(*values,c.byref(error))
        if error:
            text=message(error).decode('utf-8',errors='replace');free_error(error);raise RuntimeError(text)
        return value
    pipeline=config=profiles=profile=None;started=False
    try:
        pipeline=call('ob_create_pipeline',P,[],[])
        config=call('ob_create_config',P,[],[])
        profiles=call('ob_pipeline_get_stream_profile_list',P,[P,c.c_int],[pipeline,3])
        profile=call('ob_stream_profile_list_get_profile',P,[P,c.c_int],[profiles,0])
        pixel_format=call('ob_stream_profile_format',c.c_int,[P],[profile])
        call('ob_config_enable_stream',None,[P,P],[config,profile])
        call('ob_pipeline_start_with_config',None,[P,P],[pipeline,config]);started=True
        for _ in range(15):
            frames=depth=None
            try:
                frames=call('ob_pipeline_wait_for_frameset',P,[P,c.c_uint32],[pipeline,500])
                if not frames:continue
                depth=call('ob_frameset_depth_frame',P,[P],[frames])
                if not depth:continue
                w=call('ob_video_frame_width',c.c_uint32,[P],[depth]);h=call('ob_video_frame_height',c.c_uint32,[P],[depth])
                size=call('ob_frame_data_size',c.c_uint32,[P],[depth]);ptr=call('ob_frame_data',P,[P],[depth])
                if not w or not h or size<w*h*2:raise RuntimeError('深度帧尺寸或格式异常')
                raw=np.ctypeslib.as_array(c.cast(ptr,c.POINTER(c.c_uint16)),shape=(w*h,)).reshape(h,w).copy()
                if not np.count_nonzero(raw):continue
                scale=call('ob_depth_frame_get_value_scale',c.c_float,[P],[depth])*.001
                metres=raw.astype(np.float32)*scale
                np.save(output/'depth-raw.npy',raw);np.save(output/'depth-m.npy',metres)
                import cv2
                valid=raw>0;lo,hi=np.percentile(raw[valid],[2,98]);grey=np.clip((raw-lo)/max(1.,hi-lo)*255,0,255).astype('uint8')
                preview=cv2.applyColorMap(grey,cv2.COLORMAP_TURBO);preview[~valid]=0
                cv2.imencode('.png',preview)[1].tofile(str(output/'depth.png'))
                return dict(ok=True,status='frames_received',backend='official Orbbec SDK C API',profile_format=int(pixel_format),width=int(w),height=int(h),meters_per_unit=float(scale),valid_fraction=float(np.mean(valid)),median_distance_m=float(np.median(metres[valid])))
            finally:
                if depth:call('ob_delete_frame',None,[P],[depth])
                if frames:call('ob_delete_frame',None,[P],[frames])
        return dict(ok=False,status='no_valid_depth',message='未收到有效深度帧，请检查测距和USB连接')
    except Exception as exc:return dict(ok=False,status='sdk_error',message=str(exc))
    finally:
        if started:call('ob_pipeline_stop',None,[P],[pipeline])
        for name,ptr in [('ob_delete_stream_profile',profile),('ob_delete_stream_profile_list',profiles),('ob_delete_config',config),('ob_delete_pipeline',pipeline)]:
            if ptr:
                try:call(name,None,[P],[ptr])
                except Exception:pass
        directory.close()
