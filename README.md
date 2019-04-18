# MLFLIP

You can find the source codes for the SCA paper: [\[Um et al., 2018, "Liquid
Splash Modeling with Neural
Networks"\]](https://ge.in.tum.de/publications/2018-mlflip-um/).

## Requirements
   Activate your tensorflow environment on your machine; e.g., 
   ```
   source ~/tensorflow/bin/activate
   ```

   You need to compile the delivered mantaflow sources. Please refer to the
   general guideline to compile mantaflow: http://mantaflow.com/install.html

   MLFLIP requires a special cmake option for numpy support; e.g., in your build directory,
   ```
   cmake .. -DGUI=ON -DNUMPY=ON
   ```

## Run training simulations
   ```
   for i in {00..09}; do
     ./manta ../scenes/tsim_flip.py --nogui --seed=$i -o /tmp/tsim_flip_$i
   done
   ```


## Generate training data
   ```
   for i in {00..09}; do
     ./manta ../scenes/tdata_gen.py -o /tmp/tdata/tsim_flip_$i /tmp/tsim_flip_$i
   done
   ```


## Train a model
   ```
   ../scenes/tf_train.py --mve -o /tmp/tfmodel/ /tmp/tdata/
   ```


## Run MLFLIP
   ```
   ./manta ../scenes/mlflip.py --load /tmp/tfmodel/
   ```

   If you want to use a pre-trained model, please try:
   ```
   ./manta ../scenes/mlflip.py --load ../scenes/tfmodel/
   ```
