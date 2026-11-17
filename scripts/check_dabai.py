"""Dabai USB/driver and real RGB-D connection test; no robot or simulation imports."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/camera-check'

def worker(index):
    import cv2
    capture=cv2.VideoCapture(index,cv2.CAP_DSHOW)
    try:
        if not capture.isOpened():raise RuntimeError('无法打开指定大白彩色设备；请关闭其他占用相机的软件')
        start=time.monotonic();frame=None
        while time.monotonic()-start<8:
            ok,image=capture.read()
            if ok and image is not None:frame=image;break
        if frame is None:raise RuntimeError('设备已打开，但未收到彩色帧')
        cv2.imencode('.png',frame)[1].tofile(str(OUT/'color.png'))
        print(json.dumps(dict(ok=True,width=frame.shape[1],height=frame.shape[0])))
    finally:capture.release()

def depth_test():
    import numpy as np
    try:from pyorbbecsdk import Pipeline,Config,OBSensorType
    except ImportError:
        from dabai_sdk_depth import capture_depth
        return capture_depth(OUT)
    pipe=None;started=False
    try:
        pipe=Pipeline();config=Config()
        profiles=pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        config.enable_stream(profiles.get_default_video_stream_profile());pipe.start(config);started=True
        for _ in range(20):
            frames=pipe.wait_for_frames(500)
            frame=frames.get_depth_frame() if frames else None
            if frame:
                raw=np.frombuffer(frame.get_data(),dtype=np.uint16).reshape(frame.get_height(),frame.get_width()).copy()
                np.save(OUT/'depth-raw.npy',raw)
                if np.count_nonzero(raw):return dict(ok=True,status='frames_received',width=frame.get_width(),height=frame.get_height(),valid_fraction=float(np.mean(raw>0)))
        return dict(ok=False,status='no_valid_depth',message='未收到有效深度；请让镜头对准约0.5至1米外的非反光物体再试')
    except Exception as exc:return dict(ok=False,status='sdk_error',message=str(exc))
    finally:
        if started:pipe.stop()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--devices',type=Path);parser.add_argument('--preview',action='store_true');parser.add_argument('--rgb-worker',type=int);parser.add_argument('--depth-worker',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.rgb_worker is not None:worker(args.rgb_worker);return 0
    if args.depth_worker:print(json.dumps(depth_test(),ensure_ascii=True));return 0
    rows=json.loads(args.devices.read_text(encoding='utf-8-sig'));rows=[rows] if isinstance(rows,dict) else rows
    report=dict(devices=rows,rgb=dict(ok=False,status='not_tested'),depth=dict(ok=False,status='not_tested'))
    if not rows:
        print('未发现奥比中光 USB 设备。请检查数据线、USB口和供电。')
    else:
        print('已识别相机 USB 设备：')
        for row in rows:print(' ',row.get('Name') or '未命名设备','| 驱动错误码',row.get('ErrorCode'),'|',row['DeviceID'])
        try:
            from dshow_devices import cameras
            available=cameras();report['directshow_devices']=available
            match=[d for d in available if 'vid_2bc5' in d.get('DevicePath','').lower() or any(s in d.get('FriendlyName','').lower() for s in ['dabai','orbbec','astra'])]
            if not match:report['rgb']=dict(ok=False,status='not_enumerated',message='未在DirectShow中找到大白；不会改用笔记本摄像头代替')
            else:
                device=match[0];print('正在读取彩色画面：',device['FriendlyName'])
                done=subprocess.run([sys.executable,__file__,'--rgb-worker',str(device['index'])],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
                report['rgb']=dict(ok=done.returncode==0,device=device,details=done.stdout[-1200:],error=done.stderr[-1200:])
        except Exception as exc:report['rgb']=dict(ok=False,status='error',message=str(exc))
        depth_devices=[r for r in rows if 'PID_06' in r['DeviceID'].upper()]
        missing=[r for r in depth_devices if r.get('ErrorCode')==28]
        if missing:report['depth']=dict(ok=False,status='driver_missing',message='深度设备错误码28：尚未安装驱动，当前无法读取深度')
        elif any(r.get('ErrorCode',0)!=0 for r in depth_devices):report['depth']=dict(ok=False,status='driver_error',message='深度设备驱动异常，请查看上面的错误码')
        else:
            try:
                done=subprocess.run([sys.executable,__file__,'--depth-worker'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=25)
                report['depth']=json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode==0 else dict(ok=False,status='sdk_error',message=done.stderr[-1200:])
            except Exception as exc:report['depth']=dict(ok=False,status='error',message=str(exc))
    report['connected']=bool(rows);report['rgbd_ready']=bool(report['rgb'].get('ok') and report['depth'].get('ok'))
    (OUT/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nUSB识别：', '成功' if report['connected'] else '未发现')
    print('彩色取帧：','成功（已保存 color.png）' if report['rgb']['ok'] else '未通过，详见报告')
    print('深度取帧：','成功' if report['depth']['ok'] else report['depth'].get('message','未通过'))
    print('RGB-D可用：','是' if report['rgbd_ready'] else '尚未全部就绪')
    print('结果位置：',OUT/'result.json')
    print('驱动/SDK参考：https://github.com/orbbec/OrbbecSDK （原版大白需核对旧OpenNI协议兼容性）')
    if args.preview and report['rgb']['ok']:
        import cv2
        image=cv2.imdecode(__import__('numpy').fromfile(OUT/'color.png',dtype='uint8'),cv2.IMREAD_COLOR)
        cv2.imshow('DaBai - captured color frame (press a key to close)',image);cv2.waitKey(0);cv2.destroyAllWindows()
    return 0 if report['rgbd_ready'] else 2

if __name__=='__main__':sys.exit(main())
