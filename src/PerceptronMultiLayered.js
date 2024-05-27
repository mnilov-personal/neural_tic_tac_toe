import PlayerAwareConstraintLayer  from './PlayerAwareConstraintLayer';
import * as tf from '@tensorflow/tfjs';

export default class PerceptronMultiLayered {

    async createModel (boardPositions, outcomes, trainingEpochs) {
        try {
            // Define the ANN model
            const model = tf.sequential();        
            model.add(tf.layers.dense({ units: 9, inputShape: [9,]}));
            model.add(tf.layers.dense({ units: 9, activation: 'relu', inputShape: [9,]}));
            model.add(tf.layers.dense({ units: 2, activation: 'relu'}));
    
            tf.serialization.registerClass(PlayerAwareConstraintLayer);

            const playerAwareLayer = new PlayerAwareConstraintLayer(false);
            model.add(playerAwareLayer);
            model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    
            // Train the model
            await model.fit(boardPositions, outcomes, { epochs: trainingEpochs });

            return model;
        } catch (error) {
            console.error('Error defining ANN model:', error);
        }
    }
}