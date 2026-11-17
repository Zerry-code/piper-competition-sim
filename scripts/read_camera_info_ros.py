"""Subscribe to real ROS CameraInfo streams and save their received calibration."""
import argparse,json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import CameraInfo

class Collector(Node):
    def __init__(self,output):
        super().__init__('piper_camera_info_reader')
        self.output=output;self.output.mkdir(parents=True,exist_ok=True)
        self.subs={};self.records={};self.counts={}
        self.create_timer(.5,self.discover)
    def discover(self):
        for topic,types in self.get_topic_names_and_types():
            if 'sensor_msgs/msg/CameraInfo' in types and topic not in self.subs:
                self.subs[topic]=self.create_subscription(CameraInfo,topic,lambda msg,t=topic:self.receive(t,msg),QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT))
    def receive(self,topic,msg):
        if msg.width<=0 or msg.height<=0 or msg.k[0]<=0 or msg.k[4]<=0:return
        self.counts[topic]=self.counts.get(topic,0)+1
        value=dict(topic=topic,frame_id=msg.header.frame_id,width=msg.width,height=msg.height,fx=msg.k[0],fy=msg.k[4],cx=msg.k[2],cy=msg.k[5],distortion_model=msg.distortion_model,D=list(msg.d),K=list(msg.k),R=list(msg.r),P=list(msg.p),stamp=dict(sec=msg.header.stamp.sec,nanosec=msg.header.stamp.nanosec),samples=self.counts[topic],source='received sensor_msgs/msg/CameraInfo from running ROS camera node')
        self.records[topic]=value
        (self.output/'camera_info.json').write_text(json.dumps(self.records,ensure_ascii=False,indent=2),encoding='utf-8')
        if self.counts[topic]==1:print(json.dumps(value,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=30);args=p.parse_args()
    rclpy.init();node=Collector(args.output);start=time.monotonic()
    try:
        while time.monotonic()-start<args.timeout:
            rclpy.spin_once(node,timeout_sec=.2)
            if len(node.records)>=2 and all(v>=3 for v in node.counts.values()):break
        if not node.records:raise RuntimeError('No valid CameraInfo received; camera node/topics not ready')
        print('Received topics:',','.join(node.records),flush=True)
    finally:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
