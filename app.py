
import streamlit as st, pandas as pd, tempfile
from pathlib import Path
import plotly.express as px
from video_feature_extractor import extract_all
from master_interface import analyze

st.set_page_config(page_title="Instructional Video Intelligence",page_icon="🎓",layout="wide")
st.title("Instructional Video Intelligence")
st.caption("Upload a raw instructional video → automatic Objective 3 feature extraction → benchmark → design profile")
st.warning("Research prototype: no causal quality score or guaranteed satisfaction prediction.")

model=st.sidebar.selectbox("Whisper speech model",["tiny","base","small"],index=1)
up=st.file_uploader("Upload instructional video",type=["mp4","mov","m4v","avi","mkv"])
if up is None: st.stop()
st.video(up)
if not st.button("Analyze uploaded video",type="primary"): st.stop()

suffix=Path(up.name).suffix or ".mp4"
with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
    tmp.write(up.getbuffer()); vp=Path(tmp.name)

with st.spinner("Processing video, speech, sentiment, scenes, motion and visual characteristics..."):
    feat,transcript=extract_all(vp,model)

vals={k:feat[k] for k in ["video_length_min","avg_scene_length_min","speaking_rate_wpm","sentiment_avg","motion_avg","saturation_avg","brightness_avg","brightness_std","clarity_prop","warm_prop","flow_mag_avg"]}
rows,ds=analyze(vals,feat["instructor_face_visible"])

t1,t2,t3,t4=st.tabs(["Extracted features","Benchmark","ML design profile","Report"])
with t1:
    df=pd.DataFrame(rows)
    st.dataframe(df[["Characteristic","Value","Unit"]],use_container_width=True,hide_index=True)
    a,b,c=st.columns(3)
    a.metric("Instructor visible","Yes" if feat["instructor_face_visible"] else "No")
    b.metric("Face-detected sampled frames",f"{100*feat['face_detected_frame_prop']:.1f}%")
    c.metric("Transcript words",len(transcript.split()))
with t2:
    df=pd.DataFrame(rows); df["Standardised position"]=df["z"].clip(-2.5,2.5)
    fig=px.bar(df,x="Standardised position",y="Characteristic",orientation="h")
    fig.add_vline(x=0,line_dash="dash"); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df[["Characteristic","Value","Benchmark mean","Position"]],use_container_width=True,hide_index=True)
with t3:
    d,p=ds[0]
    st.subheader(f"Closest empirical design configuration: Profile {p['profile']}")
    st.write(p["name"])
    a,b,c=st.columns(3); a.metric("Profile size",p["n"]); b.metric("Research-profile mean rating",f"{p['rating_mean']:.2f}"); c.metric("Prototype distance",f"{d:.3f}")
    pdf=pd.DataFrame([{"Profile":q["profile"],"Name":q["name"],"Distance":x} for x,q in ds])
    st.plotly_chart(px.bar(pdf,x="Name",y="Distance"),use_container_width=True)
    st.info("Closest-profile assignment is a transparent centroid approximation to the thesis Gower–PAM profiles.")
with t4:
    out=pd.DataFrame(rows)[["Characteristic","Value","Position"]]
    st.download_button("Download analytics report",out.to_csv(index=False).encode(),"video_analytics_report.csv","text/csv")
    with st.expander("Transcript"): st.write(transcript)
