
. ./path.sh
. ./cmd.sh

set -euo pipefail
expdir=exp/train_lsm_pytorch_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs10_mli800_mlo150/decode_L2_test_beam40_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_L1
data_root=data/L2_test/CAPT-related
stage=0

. utils/parse_options.sh || exit 1;

transcript_filename=$data_root/trans_origin.txt
recog_filename=$expdir/data.json
capt_dir=$expdir/capt
# preproces
recog_TF_filename=$capt_dir/L2_test_TF.txt
# detection
testans_filename=$data_root/test_ans.txt
TR_filename=$capt_dir/TR_utt.txt
detetion_filename=$capt_dir/detection.txt
#diagnose
diagnose_filename=$capt_dir/diagnose.txt


mkdir -p $capt_dir

if [ $stage -le 0 ]; then
  echo "preprocess and get results from javascript"
  python3 local/capt/get_TF_fromjs.py --transcript_filename $transcript_filename \
                                      --recog_filename $recog_filename > $recog_TF_filename
fi

if [ $stage -le 1 ];then
  echo "detection"
  python3 local/capt/compare.py --testans_filename $testans_filename \
                                --recog_TF_filename $recog_TF_filename \
                                --TR_filename $TR_filename > $detetion_filename
fi

if [ $stage -le 2 ]; then
  echo "diagnose"
  python3 local/capt/diagnose.py --transcript_filename $transcript_filename \
                                 --recog_filename $recog_filename\
                               --TR_filename $TR_filename > $diagnose_filename
fi

echo "done";
exit 0;
