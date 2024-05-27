import logo from './logo.svg';
import './App.css';

import React from 'react';
import TicTacToe from './TicTacToe';

import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h4>Play Tic Tac Toe vs. AI Player</h4>
      </header>
      <main>
        <TicTacToe />
      </main>
    </div>
  );
}

export default App;
