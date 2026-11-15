"""EDA for the OASIS MRI slice dataset (Kaggle: kanaadlimaye/alzheimers-classification-dataset)."""
import os, hashlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT="data/oasis-mri/x/Alzheimers_Detection_dataset"; OUT="mri-eda"
NAMES={"ND":"None (ND)","VMD":"Very mild (VMD)","MD":"Mild (MD)","MoD":"Moderate (MoD)"}
order=["ND","VMD","MD","MoD"]
rows=[]
for s in ("train","valid","test"):
    d=pd.read_csv(f"{ROOT}/CSV_datafiles/_{s}_classes.csv")
    d["split"]=s; d["csv_n"]=d[order].sum(axis=1); d["csv_label"]=d[order].idxmax(axis=1).where(d.csv_n==1)
    d["label"]=d.filename.str.split("__").str[0]   # cleaned label: taken from the filename
    d["path"]=f"{ROOT}/{s}/"+d.filename
    d["src"]=d.filename.str.replace(r"_jpg\.rf\..*","",regex=True)
    rows.append(d)
df=pd.concat(rows,ignore_index=True)

ct=pd.crosstab(df.split,df.label)[order].loc[["train","valid","test"]]
fig,ax=plt.subplots(1,2,figsize=(12,4))
ct.rename(columns=NAMES).plot.bar(ax=ax[0],rot=0); ax[0].set_title("Images per class, original 4 labels"); ax[0].set_ylabel("images")
m=ct.copy(); m["MD+MoD"]=m.MD+m.MoD; m=m[["ND","VMD","MD+MoD"]]
m.rename(columns={"ND":"None","VMD":"Very mild","MD+MoD":"Mild-moderate"}).plot.bar(ax=ax[1],rot=0); ax[1].set_title("After merging Mild + Moderate (used for the model)")
plt.tight_layout(); plt.savefig(f"{OUT}/1_class_distribution.png",dpi=130); plt.close()
ct.to_csv(f"{OUT}/class_counts.csv")

fig,ax=plt.subplots(4,6,figsize=(14,10))
for i,l in enumerate(order):
    pick=df[(df.label==l)&(df.split=="train")].sample(6,random_state=1)
    for j,p in enumerate(pick.path):
        ax[i,j].imshow(Image.open(p).convert("L"),cmap="gray"); ax[i,j].axis("off")
        if j==0: ax[i,j].set_title(NAMES[l],loc="left",fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/2_sample_images.png",dpi=110); plt.close()

bad=0; sizes={}; hashes={}
for p,s in zip(df.path,df.split):
    try:
        with Image.open(p) as im: im.verify()
        with Image.open(p) as im: sizes[(im.size,im.mode)]=sizes.get((im.size,im.mode),0)+1
    except Exception: bad+=1
    hashes.setdefault(hashlib.md5(open(p,"rb").read()).hexdigest(),set()).add(s)
srcs={s:set(df[df.split==s].src) for s in ("train","valid","test")}
q=[("Total images",len(df)),("Unreadable / corrupt files",bad),("Image size and mode",str(sizes)),
   ("CSV rows with all-zero labels (fixed using filename label)",int((df.csv_n==0).sum())),
   ("CSV label disagrees with filename label (filename used)",int((df.csv_label.notna()&(df.csv_label!=df.label)).sum())),
   ("Identical files found in more than one split",sum(len(v)>1 for v in hashes.values())),
   ("Source images shared train/valid",len(srcs["train"]&srcs["valid"])),
   ("Source images shared train/test",len(srcs["train"]&srcs["test"])),
   ("Train: images / unique source images",f"{(df.split=='train').sum()} / {len(srcs['train'])}")]
qd=pd.DataFrame(q,columns=["check","result"]); qd.to_csv(f"{OUT}/3_quality_checks.csv",index=False)
print(qd.to_string(index=False)); print(ct)

samp=df[df.split=="train"].groupby("label").sample(80,random_state=0)
mi,br=[],[]
for p in samp.path:
    a=np.asarray(Image.open(p).convert("L"),dtype=float)/255
    mi.append(a.mean()); br.append((a>0.15).mean())
samp=samp.assign(mean_int=mi,tissue=br)
fig,ax=plt.subplots(1,2,figsize=(12,4))
for k,col,t in [(0,"mean_int","Average brightness per scan"),(1,"tissue","Share of scan covered by tissue")]:
    ax[k].boxplot([samp[samp.label==l][col] for l in order],tick_labels=[NAMES[l] for l in order]); ax[k].set_title(t); ax[k].tick_params(axis="x",rotation=15)
plt.tight_layout(); plt.savefig(f"{OUT}/4_intensity_and_tissue.png",dpi=130); plt.close()
print(samp.groupby("label")[["mean_int","tissue"]].mean().round(3))
