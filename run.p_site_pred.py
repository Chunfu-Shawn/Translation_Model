import os
import time
from model.p_site_predictor_lightgbm_onehot import PSitePredictor


lib_path = "/home/user/data3/rbase/translation_model/models/lib"
data_path = "/home/user/data3/yaoc/translation_model/data/read_count/filtered"
result_dir = "/home/user/data3/rbase/translation_model/results/p_site_pred"

# initiation
preditor = PSitePredictor(
    tree_index_file=os.path.join(lib_path, "genome_index_tree.pkl"),
    transcript_seq_file=os.path.join(lib_path, 'tx_seq.v48.pkl'),
    transcript_meta_file=os.path.join(lib_path, 'transcript_meta.pkl'),
    transcript_cds_file=os.path.join(lib_path, 'transcript_cds.pkl'),
    n_thread = 60
)

samples = ["Huh7", "fat_tissue"]
endonucleases = ["MNase", "RNase I"]

for idx, sample in enumerate(samples):
    output_dir = os.path.join(result_dir, sample, "lightgbm")

    # if no done results
    if not os.path.exists(os.path.join(output_dir, "p_site_count.pkl")):
        print(f"### Predict P site for sample: {sample} ({endonucleases[idx]}) ###")
        # load count data
        preditor.load_read_count_data(
            os.path.join(data_path, sample + ".read_count.pkl"),
            endonuclease = endonucleases[idx])
        time_s = time.time()
        # train model
        preditor.train(
            output_dir
        )
        time_e = time.time()
        print(f"[TIMING] train={time_e-time_s:.4f}s")
        # predict p site
        p_site_dict = preditor.predict(
            output_dir,
            batch_transcripts = 5000
        )
        time_p = time.time()
        print(f"[TIMING] pred={time_p-time_e:.4f}s")
        # eval p site
        preditor.eval_p_site(
            p_site_dict,
            output_dir
            )
    else:
        print(f"### Predicting P site for sample {sample} have done ###")