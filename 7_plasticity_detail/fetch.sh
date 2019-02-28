# scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/teacher_plasticity .
# scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/no_teacher_plasticity .
# scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/teacher_no_plasticity .
# scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/no_teacher_no_plasticity .
#scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/teacher_plasticity_nostim .
scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/teacher_experiment1 .
scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/teacher_experiment2 .
scp -r nemo:/work/ws/nemo/fr_ib131-my_ws-0/mnist-nest/6_mnist_with_plasticity/teacher_experiment3 .
# python3 data_processing.py --name no_teacher_no_plasticity
# python3 data_processing.py --name no_teacher_plasticity
# python3 data_processing.py --name teacher_no_plasticity
# python3 data_processing.py --name teacher_plasticity
python3 data_processing.py --name teacher_experiment1
python3 data_processing.py --name teacher_experiment2
python3 data_processing.py --name teacher_experiment3