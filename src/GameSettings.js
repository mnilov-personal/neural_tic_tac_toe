import React, { useState } from 'react';
import { TextField, Button, Grid, Typography } from '@mui/material';
import { styled } from '@mui/system';

const FormContainer = styled('div')(({ theme }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
}));

const FormField = styled(TextField)(({ theme }) => ({
  marginBottom: theme.spacing(2),
}));

const SubmitButton = styled(Button)(({ theme }) => ({
  marginTop: theme.spacing(2),
}));

const GameSettings = ({ onData , formData}) => {
  var [epochs, setEpochs] = useState(formData.trainFor);

  const handleSubmit = (event) => {
    event.preventDefault();
    onData(epochs);
  };

  return (
    <Grid container direction="column" alignItems="center">
      <Typography variant="h4" gutterBottom>
        Perceptron Configuration
      </Typography>
      <FormContainer>
        <form onSubmit={handleSubmit}>
          <FormField
            label="Training Epochs"
            type="number"
            variant="outlined"
            value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            fullWidth
            required
          />
          <SubmitButton
            type="submit"
            variant="contained"
            color="primary"
          >
            Apply
          </SubmitButton>
        </form>
      </FormContainer>
    </Grid>
  );
};


export default GameSettings;