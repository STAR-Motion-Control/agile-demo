#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless GR00T-WBC stepping-limit sweep on scene_29dof (no DDS, CPU onnx).

Reproduces run_mujoco_gear_wbc.py's obs/PD/policy EXACTLY (same Balance/Walk
onnx, same yaml gains/scales) so results transfer to the real adapter. Drives a
scripted (vx,vy,wz) command for a duration from a settled stand, then measures
net base displacement / stepping. Answers: what is the minimum velocity x
duration that produces a real committed step (not just waist sway)?
"""
import os, sys, json, argparse, collections
import numpy as np
import mujoco
import onnxruntime as ort

HOME = os.path.expanduser("~")
XML = f"{HOME}/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml"
POLDIR = f"{HOME}/GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
BAL = f"{POLDIR}/GR00T-WholeBodyControl-Balance.onnx"
WALK = f"{POLDIR}/GR00T-WholeBodyControl-Walk.onnx"

# --- config from g1_gear_wbc.yaml (ground truth) ---
KPS = np.array([150,150,150,200,40,40, 150,150,150,200,40,40, 250,250,250], np.float32)
KDS = np.array([2,2,2,4,2,2, 2,2,2,4,2,2, 5,5,5], np.float32)
DEFAULT = np.array([-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0, 0,0,0], np.float32)
ANG_VEL_SCALE=0.5; DOF_POS_SCALE=1.0; DOF_VEL_SCALE=0.05; ACTION_SCALE=0.25
CMD_SCALE=np.array([2.0,2.0,0.5], np.float32)
SIM_DT=0.005; DECIM=4; NUM_ACT=15; N_JOINTS=29; OBS_HL=6; SOBS=86
HEIGHT_STAND=0.74; WALK_THRESH=0.05
ARM_KP=100.0; ARM_KD=0.5

def quat_rot_inv(q, v):
    w,x,y,z = q; qc=np.array([w,-x,-y,-z])
    return np.array([
        v[0]*(qc[0]**2+qc[1]**2-qc[2]**2-qc[3]**2)+v[1]*2*(qc[1]*qc[2]-qc[0]*qc[3])+v[2]*2*(qc[1]*qc[3]+qc[0]*qc[2]),
        v[0]*2*(qc[1]*qc[2]+qc[0]*qc[3])+v[1]*(qc[0]**2-qc[1]**2+qc[2]**2-qc[3]**2)+v[2]*2*(qc[2]*qc[3]-qc[0]*qc[1]),
        v[0]*2*(qc[1]*qc[3]-qc[0]*qc[2])+v[1]*2*(qc[2]*qc[3]+qc[0]*qc[1])+v[2]*(qc[0]**2-qc[1]**2-qc[2]**2+qc[3]**2),
    ])

def grav_orient(q): return quat_rot_inv(q, np.array([0.,0.,-1.]))

def yaw_from_quat(q):
    w,x,y,z=q
    import math
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = SIM_DT
        so = ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
        prov=["CPUExecutionProvider"]
        self.bal = ort.InferenceSession(BAL, sess_options=so, providers=prov)
        self.walk = ort.InferenceSession(WALK, sess_options=so, providers=prov)
        self.bal_in = self.bal.get_inputs()[0].name
        self.walk_in = self.walk.get_inputs()[0].name
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lfoot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.rfoot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.reset_stand()

    def reset_stand(self):
        mujoco.mj_resetData(self.m, self.d)
        # try a keyframe first
        if self.m.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        else:
            self.d.qpos[:] = 0
            self.d.qpos[2] = 0.793
            self.d.qpos[3:7] = [1,0,0,0]
            self.d.qpos[7:7+NUM_ACT] = DEFAULT
        mujoco.mj_forward(self.m, self.d)
        self.action = np.zeros(NUM_ACT, np.float32)
        self.hist = collections.deque([np.zeros(SOBS,np.float32)]*OBS_HL, maxlen=OBS_HL)
        self.obs = np.zeros(SOBS*OBS_HL, np.float32)

    def compute_obs(self, loco, height):
        d=self.d
        command=np.zeros(7,np.float32)
        command[:3]=np.asarray(loco,np.float32)*CMD_SCALE
        command[3]=height
        qj=d.qpos[7:7+N_JOINTS].copy(); dqj=d.qvel[6:6+N_JOINTS].copy()
        quat=d.qpos[3:7].copy(); omega=d.qvel[3:6].copy()
        pad=np.zeros(N_JOINTS,np.float32); pad[:NUM_ACT]=DEFAULT
        qj_s=(qj-pad)*DOF_POS_SCALE; dqj_s=dqj*DOF_VEL_SCALE
        grav=grav_orient(quat); omega_s=omega*ANG_VEL_SCALE
        so=np.zeros(SOBS,np.float32)
        so[0:7]=command; so[7:10]=omega_s; so[10:13]=grav
        so[13:13+N_JOINTS]=qj_s; so[13+N_JOINTS:13+2*N_JOINTS]=dqj_s
        so[13+2*N_JOINTS:13+2*N_JOINTS+15]=self.action
        return so

    def control(self, loco, height):
        so=self.compute_obs(loco, height)
        self.hist.append(so)
        for i,h in enumerate(self.hist): self.obs[i*SOBS:(i+1)*SOBS]=h
        inp=self.obs[None].astype(np.float32)
        if np.linalg.norm(np.asarray(loco)) <= WALK_THRESH:
            self.action=self.bal.run(None,{self.bal_in:inp})[0].squeeze().astype(np.float32)
        else:
            self.action=self.walk.run(None,{self.walk_in:inp})[0].squeeze().astype(np.float32)
        self.target = self.action*ACTION_SCALE + DEFAULT

    def step_once(self, loco, height):
        # one control tick = DECIM sim steps
        self.control(loco, height)
        for _ in range(DECIM):
            d=self.d
            leg_tau=(self.target - d.qpos[7:7+NUM_ACT])*KPS + (0 - d.qvel[6:6+NUM_ACT])*KDS
            d.ctrl[:NUM_ACT]=leg_tau
            arm_tau=(0 - d.qpos[7+NUM_ACT:7+N_JOINTS])*ARM_KP + (0 - d.qvel[6+NUM_ACT:6+N_JOINTS])*ARM_KD
            d.ctrl[NUM_ACT:]=arm_tau
            mujoco.mj_step(self.m, self.d)

    def base(self): return self.d.xpos[self.pelvis].copy()
    def footz(self): return self.d.xpos[self.lfoot][2], self.d.xpos[self.rfoot][2]

def run_case(sim, vx, vy, wz, T_move, height=HEIGHT_STAND, warmup=2.5, settle=3.0,
             pre_vx=0.0, pre_T=0.0, traj=False):
    """Settle standing, (optional pre-step warmup), move for T_move, settle. Return metrics."""
    sim.reset_stand()
    ctrl_dt = SIM_DT*DECIM
    # settle standing under Balance
    for _ in range(int(warmup/ctrl_dt)):
        sim.step_once([0,0,0], height)
    x0,y0,_=sim.base(); yaw0=yaw_from_quat(sim.d.qpos[3:7])
    tr=[]
    lz0,rz0=sim.footz(); footbase=min(lz0,rz0)
    xs=[]; foot_lifts=0; lift_state=[False,False]
    fell=False
    # optional warm-up pre-step (a fixed larger command to spin up the gait)
    for _ in range(int(pre_T/ctrl_dt)):
        sim.step_once([pre_vx,0,0], height)
        xs.append(sim.base()[0]-x0)
    # the actual test move
    nmove=int(round(T_move/ctrl_dt))
    peak_move=0.0
    for k in range(nmove):
        sim.step_once([vx,vy,wz], height)
        b=sim.base(); dx=b[0]-x0; xs.append(dx); peak_move=max(peak_move,dx)
        if traj: tr.append((round((k+1)*ctrl_dt,3), round(dx*100,2)))
        lz,rz=sim.footz()
        for i,fz in enumerate((lz,rz)):
            up=fz-footbase>0.020
            if up and not lift_state[i]: foot_lifts+=1
            lift_state[i]=up
        if b[2]<0.40: fell=True
    dx_at_cmd_end=xs[-1] if xs else 0.0
    # settle (command zero -> Balance) and let it come to rest
    for k in range(int(settle/ctrl_dt)):
        sim.step_once([0,0,0], height)
        b=sim.base(); dx=b[0]-x0; xs.append(dx)
        if traj: tr.append((round(T_move+(k+1)*ctrl_dt,3), round(dx*100,2)))
        lz,rz=sim.footz()
        for i,fz in enumerate((lz,rz)):
            up=fz-footbase>0.020
            if up and not lift_state[i]: foot_lifts+=1
            lift_state[i]=up
        if b[2]<0.40: fell=True
    xf,yf,_=sim.base(); yawf=yaw_from_quat(sim.d.qpos[3:7])
    xs=np.array(xs)
    return dict(vx=vx,vy=vy,wz=wz,T=T_move,height=height,pre_vx=pre_vx,pre_T=pre_T,
                net_dx=float(xf-x0), net_dy=float(yf-y0), net_dyaw=float(yawf-yaw0),
                cmd_dist=float(vx*T_move), realized=float((xf-x0)/(vx*T_move)) if vx*T_move>1e-9 else 0.0,
                min_dx=float(xs.min()), max_dx=float(xs.max()), peak_move=float(peak_move),
                dx_at_cmd_end=float(dx_at_cmd_end), settle_back=float(peak_move-(xf-x0)),
                foot_lifts=int(foot_lifts), fell=bool(fell), pelvis_z=float(sim.base()[2]),
                traj=tr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",default="smoke",choices=["smoke","sweep","warmup","height","traj","sens","yaw","warmrobust","hcut","calib","senslong"])
    ap.add_argument("--out",default="")
    a=ap.parse_args()
    sim=Sim()
    print(f"model nq={sim.m.nq} nu={sim.m.nu} nkey={sim.m.nkey}", file=sys.stderr)
    results=[]
    if a.mode=="smoke":
        for vx,T in [(0.20,2.0),(0.08,0.6),(0.12,0.6),(0.10,1.0)]:
            r=run_case(sim,vx,0,0,T)
            results.append(r)
            print(f"vx={vx:.2f} T={T:.1f} -> net_dx={r['net_dx']*100:+6.1f}cm "
                  f"cmd={r['cmd_dist']*100:5.1f}cm realized={r['realized']*100:5.0f}% "
                  f"lifts={r['foot_lifts']} min_dx={r['min_dx']*100:+.1f} fell={r['fell']} z={r['pelvis_z']:.2f}")
    elif a.mode=="sweep":
        vxs=[0.08,0.10,0.12,0.15,0.20,0.25,0.30]
        Ts=[0.6,0.8,1.0,1.2,1.5,2.0,2.5,3.0]
        for vx in vxs:
            for T in Ts:
                r=run_case(sim,vx,0,0,T); results.append(r)
                print(f"vx={vx:.2f} T={T:.1f} net={r['net_dx']*100:+6.1f}cm "
                      f"peak={r['peak_move']*100:+5.1f} back={r['min_dx']*100:+5.1f} "
                      f"sback={r['settle_back']*100:4.1f} cmd={r['cmd_dist']*100:5.1f} "
                      f"real={r['realized']*100:4.0f}% lifts={r['foot_lifts']:2d} fell={int(r['fell'])}")
    elif a.mode=="traj":
        for vx,T in [(0.15,1.5),(0.20,1.5),(0.10,0.6),(0.20,3.0)]:
            r=run_case(sim,vx,0,0,T,traj=True); results.append(r)
            print(f"\n### vx={vx} T={T}  net={r['net_dx']*100:+.1f}cm peak={r['peak_move']*100:+.1f} back={r['min_dx']*100:+.1f}")
            print("  t(s):dx(cm) ", " ".join(f"{t}:{d:+.1f}" for t,d in r['traj'][::3]))
    elif a.mode=="warmup":
        # compare: plain short move vs same move preceded by a 1-step warm-up
        for vx,T in [(0.08,0.6),(0.10,0.6),(0.12,0.6),(0.10,0.8)]:
            r0=run_case(sim,vx,0,0,T,pre_vx=0.0,pre_T=0.0)
            r1=run_case(sim,vx,0,0,T,pre_vx=0.15,pre_T=0.4)
            results+= [r0,r1]
            print(f"vx={vx:.2f} T={T:.1f} plain net={r0['net_dx']*100:+5.1f}cm lifts={r0['foot_lifts']} | "
                  f"+warmup(0.15,0.4s) net={r1['net_dx']*100:+5.1f}cm lifts={r1['foot_lifts']}")
    elif a.mode=="height":
        for h in [0.74,0.62,0.55,0.50]:
            for vx,T in [(0.12,1.0),(0.15,1.0)]:
                r=run_case(sim,vx,0,0,T,height=h); results.append(r)
                print(f"h={h:.2f} vx={vx:.2f} T={T:.1f} net_dx={r['net_dx']*100:+6.1f}cm "
                      f"real={r['realized']*100:4.0f}% lifts={r['foot_lifts']:2d} fell={int(r['fell'])} z={r['pelvis_z']:.2f}")
    elif a.mode=="warmrobust":
        # does a warm-up pre-step STABILIZE the sign of a short move across sway phases?
        for vx,T in [(0.10,0.6),(0.12,0.6)]:
            plain=[]; warm=[]
            for wu in [1.0,1.5,2.0,2.5,3.0,3.5]:
                plain.append(run_case(sim,vx,0,0,T,warmup=wu)['net_dx']*100)
                warm.append(run_case(sim,vx,0,0,T,warmup=wu,pre_vx=0.15,pre_T=0.4)['net_dx']*100)
            print(f"vx={vx:.2f} T={T:.1f}")
            print(f"   plain net by phase: "+" ".join(f"{v:+5.1f}" for v in plain)+
                  f"  [min {min(plain):+.1f}, max {max(plain):+.1f}]")
            print(f"   +warmup   by phase: "+" ".join(f"{v:+5.1f}" for v in warm)+
                  f"  [min {min(warm):+.1f}, max {max(warm):+.1f}]")
    elif a.mode=="hcut":
        for h in [0.74,0.72,0.70,0.68,0.66,0.64,0.62]:
            r=run_case(sim,0.15,0,0,1.5,height=h)
            print(f"h={h:.2f} vx=0.15 T=1.5 net_dx={r['net_dx']*100:+6.1f}cm lifts={r['foot_lifts']:2d} z={r['pelvis_z']:.2f}")
    elif a.mode=="calib":
        # target D cm at cruise vx=0.15 with T=(D/vx+23)/71 (fit); check net vs target across phases
        for D in [8.0,12.0,18.0]:
            vx=0.15; T=(D/(vx*100)+0.23)/0.71   # D cm -> m; fit net[m]=vx*(0.71*T-0.23)
            nets=[run_case(sim,vx,0,0,T,warmup=wu)['net_dx']*100 for wu in [1.5,2.0,2.5,3.0]]
            print(f"target={D:.0f}cm -> vx={vx} T={T:.2f}s  net across phases: "+
                  " ".join(f"{v:+5.1f}" for v in nets)+f"  (mean {sum(nets)/len(nets):+.1f})")
    elif a.mode=="senslong":
        for vx,T in [(0.12,1.2),(0.12,1.5),(0.12,2.0),(0.15,1.2),(0.15,1.5),(0.20,1.2)]:
            row=[run_case(sim,vx,0,0,T,warmup=wu)['net_dx']*100 for wu in [1.0,1.5,2.0,2.5,3.0,3.5]]
            print(f"vx={vx:.2f} T={T:.1f}  net by sway-phase: "+" ".join(f"{v:+5.1f}" for v in row)+
                  f"   [min {min(row):+.1f} max {max(row):+.1f} #neg {sum(1 for v in row if v<0)}]")
    elif a.mode=="sens":
        # how fragile is a short command to how settled the robot is at move-start?
        for vx,T in [(0.10,0.6),(0.12,0.8),(0.15,1.0)]:
            row=[]
            for wu in [1.0,1.5,2.0,2.5,3.5]:
                r=run_case(sim,vx,0,0,T,warmup=wu); row.append((wu,r['net_dx']*100))
            print(f"vx={vx:.2f} T={T:.1f}  net_dx by warmup(s): "+
                  " ".join(f"{wu:.1f}s={n:+5.1f}cm" for wu,n in row))
    elif a.mode=="yaw":
        for wz in [0.10,0.12,0.15,0.20,0.30]:
            for T in [0.6,1.0,1.5,2.0]:
                r=run_case(sim,0,0,wz,T); results.append(r)
                import math
                print(f"wz={wz:.2f} T={T:.1f} net_dyaw={math.degrees(r['net_dyaw']):+6.1f}deg "
                      f"cmd={math.degrees(wz*T):5.1f} real={math.degrees(r['net_dyaw'])/max(1e-6,math.degrees(wz*T))*100:4.0f}% "
                      f"lifts={r['foot_lifts']:2d} fell={int(r['fell'])}")
    if a.out:
        with open(a.out,"w") as f: json.dump(results,f,indent=2)
        print("wrote",a.out,file=sys.stderr)

if __name__=="__main__":
    main()
