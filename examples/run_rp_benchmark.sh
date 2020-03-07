python ./examples/rp_benchmark.py -model mf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search random -batch_size 1024
python ./examples/rp_benchmark.py -model mf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search bayesian -batch_size 1024
python ./examples/rp_benchmark.py -model mf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search hyperband -batch_size 1024

python ./examples/rp_benchmark.py -model mlp -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search random -batch_size 1024
python ./examples/rp_benchmark.py -model mlp -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search bayesian -batch_size 1024
python ./examples/rp_benchmark.py -model mlp -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search hyperband -batch_size 1024

python ./examples/rp_benchmark.py -model gmf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search random -batch_size 1024
python ./examples/rp_benchmark.py -model gmf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search bayesian -batch_size 1024
python ./examples/rp_benchmark.py -model gmf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search hyperband -batch_size 1024

python ./examples/rp_benchmark.py -model neumf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search random -batch_size 1024
python ./examples/rp_benchmark.py -model neumf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search bayesian -batch_size 1024
python ./examples/rp_benchmark.py -model neumf -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search hyperband -batch_size 1024

python ./examples/rp_benchmark.py -model autorec -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search random -batch_size 1024
python ./examples/rp_benchmark.py -model autorec -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search bayesian -batch_size 1024
python ./examples/rp_benchmark.py -model autorec -data ml -data_path ./examples/datasets/ml-1m/ratings.dat -search hyperband -batch_size 1024