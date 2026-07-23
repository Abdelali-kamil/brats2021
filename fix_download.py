import tcia_utils.nbia as nbia

# This downloads the NIfTI collection
nbia.downloadSeries(
    collection="UPENN-GBM",
    format="NIfTI", 
    downloadDir="/home/kamilabdelali/brats2021/upenn_nifti"
)