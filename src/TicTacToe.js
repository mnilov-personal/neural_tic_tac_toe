import React from 'react';
import './TicTacToe.css';
import * as tf from '@tensorflow/tfjs';
import PerceptronMultiLayered from './PerceptronMultiLayered';
import GameSettings from './GameSettings';
import styled from 'styled-components';
import { Box, Paper, Button } from '@mui/material';

// Training data series - opponent positions
const xs = tf.tensor2d( [[1, 0, 0, 
                          1, 0, 0, 
                          0, 0, 0],

                          [1, 0, 0, 
                          0, 0, 0, 
                          1, 0, 0],

                          [1, 0, 0,
                          0, 0, 0, 
                          0, 0, 1],

                          [0, 0, 1, 
                          0, 0, 0, 
                          1, 0, 0],

                          [1, 0, 1, 
                          0, 0, 0, 
                          0, 0, 0],

                          [0, 0, 0, 
                          1, 0, 1, 
                          0, 0, 0],

                          [0, 0, 0, 
                          0, 0, 0, 
                          1, 0, 1],

                          [0, 0, 0, 
                          0, 0, 1, 
                          1, 0, 0],

                          [1, 0, 0, 
                          0, 0, 1, 
                          0, 0, 0]
                      ]);

// Training data series - machine moves
const ys = tf.tensor2d([[1,3],                       
                        [1,2],
                        [2,2],
                        [2,2],
                        [1,2],
                        [2,2],
                        [3,2],
                        [1,1],
                        [3,2]
                    ]);

const ResetButton = styled(Button)``;

const BoardRow = styled(Box)`
  display: flex;
`;

const Square = styled(Paper)`
          font-size: 36px;
          width: 60px;
          height: 60px;
          margin-right: -1px;
          margin-bottom: -1px;
          display: flex;
          justify-content: center;
          align-items: center;
          background-color: #fff;
          border: 1px solid #999;
          &:hover {
            background-color: #f0f0f0;
            cursor: pointer;
          }
`;

class Board extends React.Component {
  renderSquare(i) {
    return (
      <Square elevation={3} onClick={() => this.props.onClick(i)}>
        {this.props.squares[i]}
      </Square>
    );
  }

  render() {
    return (
      <div>
        <BoardRow>
          {this.renderSquare(0)}
          {this.renderSquare(1)}
          {this.renderSquare(2)}
        </BoardRow>
        <BoardRow>
          {this.renderSquare(3)}
          {this.renderSquare(4)}
          {this.renderSquare(5)}
        </BoardRow>
        <BoardRow>
          {this.renderSquare(6)}
          {this.renderSquare(7)}
          {this.renderSquare(8)}
        </BoardRow>
      </div>
    );
  }
}

class TicTacToe extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      squares: Array(9).fill(null),
      model: null,
      isTraining: true,
      activeTab: 'game',
      trainingEpochs: 80,
      winner: null
    };
    this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
  }

  handleSettingsUpdate(noEpochs) {
    this.setState({ trainingEpochs: noEpochs });
    this.setState({ isTraining: true });
    const perceptronModel = new PerceptronMultiLayered(this.state.trainingEpochs);
    perceptronModel.createModel(xs, ys, this.state.trainingEpochs).then(model => {
      this.setState({ model });
      this.setState({ isTraining: false });
    }).catch(error => {
      console.error(error);
    });
  }

  handleTabChange(tab){
    this.setState({ activeTab: tab });
  };

  updatePlayerPosition(newPosition) {
    const constraintLayer = this.state.model.layers[3];
    constraintLayer.setPlayerPosition(newPosition);
  }

  async componentDidMount() {
    this.setState({ isTraining: true });
    const perceptronModel = new PerceptronMultiLayered(this.state.trainingEpochs);
    perceptronModel.createModel(xs, ys, this.state.trainingEpochs).then(model => {
      this.setState({ model });
      this.setState({ isTraining: false });
    }).catch(error => {
      console.error(error);
    });
  }

  componentWillUnmount() {
    if (this.state.model) {
      this.state.model.dispose();
    }
  }

  machineMove(squares) {
    const humanPositions = squares.map(s=>(s === "X") ? 1 : 0);
    const humanPositionsTensor = tf.tensor2d(humanPositions, [1,9]);
    var board_positions_tensor = tf.tensor1d(squares.map(s=>(s === "X")||(s === "O") ? 1 : 0));
    this.updatePlayerPosition(board_positions_tensor);
    const predictions = this.state.model.predict(humanPositionsTensor);
    return predictions.dataSync();
  }

  insertXAtIndex(x, y, array) {
    const index = (x * 3) + y;
    array[index] = 'O';
    return array;
  }

  handleClick(i) {
    const squares = this.state.squares.slice();
    if (squares[i] || this.state.winner)
      return;
      
    squares[i] = 'X';

    if (calculateWinner(squares)) {
      this.setState({
        squares: squares,
        winner : 'X'
      });
      return;
    }

    // ANN produces machine move
    let aiResult = this.machineMove(squares);

    // Insert O as machine
    this.insertXAtIndex(aiResult[0], aiResult[1], squares);

    this.setState({
      squares: squares,
      winner : calculateWinner(squares)
    });
  }

  handleReset = () => {
    this.setState({
      squares: Array(9).fill(null),
      winner : null
    });
  };

  render() {
    const { activeTab , trainingEpochs, winner, isTraining} = this.state;

    let status;

    if (winner) {
      status = 'Winner: ' + winner;
    } else {
      status = 'Next player: X';
    }
 
    return (
        <div>
          <div className="app">
            <div>
              <p>
              {isTraining ? ('Training AI player ..') : (<>&nbsp;</>)}
              </p>
            </div>
            <div className="tabs">
              <button onClick={() => this.handleTabChange('game')} className={activeTab === 'game' ? 'active' : ''}>Game</button>
              <button onClick={() => this.handleTabChange('settings')} className={activeTab === 'settings' ? 'active' : ''}>Settings</button>
            </div>
            <div className="tab-content">
              {activeTab === 'settings' && <GameSettings onData={this.handleSettingsUpdate} formData={{trainFor : trainingEpochs}}/>}
              {activeTab === 'game' && 
                <>
                  {!isTraining && (
                    <>
                    <div className="status">{status}</div>
                    <div className="game-board">
                    <Board
                      squares={this.state.squares}
                      onClick={(i) => this.handleClick(i)}
                    />
                    </div>
                    <div className='resetButton'>
                      <ResetButton onClick={this.handleReset}
                                  variant="contained"
                                  color="primary"
                      >
                        New Game
                      </ResetButton>
                    </div>
                    </>
                  )}
                </>         
              }
            </div>
          </div>
        </div>
    );
  }
}

// Function that determines the winner
function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}

export default TicTacToe;