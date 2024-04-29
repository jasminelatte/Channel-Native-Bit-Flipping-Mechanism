```


sh run_d2d_fedavg_standalone_pytorch.sh 0 20 fed_cifar100 ./data/fed_cifar100/datasets resnet18_gn 30 50 0.1 sgd 10 fed_cifar100_10_client.h5 4 2265 7.55 0 0 0.98
```

```
Gaussian_indicator: 0 1 2 3
'3': # bit flipping; only artificial noise for protecting privacy
'1': #communication noise: Bit flipping; Artifical noise: Gaussain Noise
'2': #Gaussain Mechanism and aggregate with all received model.
# superimposed artificial and communication noise for bit flipping
```