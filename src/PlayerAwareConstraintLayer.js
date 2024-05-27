import * as tf from '@tensorflow/tfjs';

export default class PlayerAwareConstraintLayer extends tf.layers.Layer {
    constructor(trainMe) {
        super({});
        this.trainable = trainMe;
        this.playerPosition = null;
    }
  
    findClosestUnpopulatedNode = (gridArray, x, y) => {
      const rows = 3;
      const cols = 3;
      let minDistance = Infinity;
      let closestNode = null;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (gridArray[i * cols + j] === 0) {
            const distance = Math.abs((i - x) * (j - y));
            if (distance < minDistance) {
              minDistance = distance;
              closestNode = [i,j];
            }
          }
        }
      }
    
      return closestNode;
    };
  
    placeXInGrid = (tensor, x, y) => {
      const closestNode = this.findClosestUnpopulatedNode(tensor, x, y);
      const retValue = (closestNode) ? (closestNode) :  null;
  
      return retValue;
    };
  
    computeOutputShape(inputShape) {
      return inputShape;
    }
  
    build(inputShape) {
        // This layer does not build any weights
        this.built = true;
    }
  
    getClassName() {
      return 'PlayerAwareConstraintLayer';
    }
  
    substituteFirstRow(tensor2D, tensor1D) {
      if (tensor2D.shape[1] !== tensor1D.shape[0]) {
          throw new Error("The number of elements in the 1D tensor must match the number of columns in the 2D tensor.");
      }
      const tail = tensor2D.slice([1, 0], [tensor2D.shape[0] - 1, tensor2D.shape[1]]);
      return tf.concat([tensor1D.reshape([1, tensor1D.shape[0]]), tail], 0);
  }
  
    call(inputs, kwargs) {
      const isTraining = kwargs.training || false;
      const roundedInput = tf.step(inputs[0]);
  
      // Do not push predicted position away from taken positions on game board when training
      if (!isTraining) {
  
        inputs[0] = roundedInput;
        const firstRow = inputs[0].gather(0);
        const x = firstRow.dataSync()[0];
        const y = firstRow.dataSync()[1];
        const corrected = this.placeXInGrid(this.playerPosition.arraySync(), x, y);
   
        return (corrected) ? this.substituteFirstRow(inputs[0], tf.tensor1d(corrected)) : inputs;
      } else {
        return inputs;
      }
    }
  
    getConfig() {
        return {
            playerPosition: this.playerPosition.arraySync(),
            trainable: this.trainable
        };
    }
  
    setPlayerPosition(inTensor){
      this.playerPosition = inTensor;
    }
  
    static className = 'PlayerAwareConstraintLayer';
    static classFunc = (config) => new PlayerAwareConstraintLayer(config.playerPosition, config.trainable);
  }