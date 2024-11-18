import React, { useState } from 'react';

const ConfusionMatrix = () => {
  // Confusion matrix data for each model with accuracy
  const confMatrixData = {
    linearRegression: {
      accuracy: 98.62,
      data: [
        { trueLabel: 'Caution', predictedLabels: [4337, 0, 0, 0] },
        { trueLabel: 'Extreme Caution', predictedLabels: [200, 139263, 500, 402] },
        { trueLabel: 'Danger', predictedLabels: [0, 0, 668, 100] },
        { trueLabel: 'Extreme Danger', predictedLabels: [0, 0, 0, 100] },
      ]
    },
    knn: {
      accuracy: 98.62,
      data: [
        { trueLabel: 'Caution', predictedLabels: [3740, 50, 0, 0] },
        { trueLabel: 'Extreme Caution', predictedLabels: [100, 139190, 435, 265] },
        { trueLabel: 'Danger', predictedLabels: [0, 646, 467, 254] },
        { trueLabel: 'Extreme Danger', predictedLabels: [0, 0, 0, 163] },
      ]
    },
    decisionTree: {
      accuracy: 97.51,
      data: [
        { trueLabel: 'Caution', predictedLabels: [528, 0, 0, 0] },
        { trueLabel: 'Extreme Caution', predictedLabels: [1582, 128833, 550, 450] },
        { trueLabel: 'Danger', predictedLabels: [0, 1043, 12584, 0] },
        { trueLabel: 'Extreme Danger', predictedLabels: [0, 0, 0, 0] },
      ]
    },
    randomForest: {
      accuracy: 97.95,
      data: [
        { trueLabel: 'Caution', predictedLabels: [6524, 0, 0, 0] },
        { trueLabel: 'Extreme Caution', predictedLabels: [1392, 131047, 184, 0] },
        { trueLabel: 'Danger', predictedLabels: [0, 1316, 5015, 0] },
        { trueLabel: 'Extreme Danger', predictedLabels: [0, 0, 92, 0] },
      ]
    }
  };

  const [selectedModel, setSelectedModel] = useState('linearRegression');

  // Calculate total correct predictions and total predictions
  const getTotalPredictions = (data) => data.reduce((sum, row) => sum + row.predictedLabels.reduce((a, b) => a + b, 0), 0);
  const getTotalCorrectPredictions = (data) => data.reduce((sum, row, index) => sum + row.predictedLabels[index], 0);

  const modelData = confMatrixData[selectedModel];
  const totalPredictions = getTotalPredictions(modelData.data);
  const totalCorrectPredictions = getTotalCorrectPredictions(modelData.data);

  // Function to determine dynamic styles based on value
  const getStyleForValue = (value) => {
    if (value > 100000) return 'bg-orange-800'; // Dark Orange
    if (value > 10000) return 'bg-orange-600'; // Medium Orange
    if (value > 100) return 'bg-orange-400'; // Light Orange
    return 'bg-orange-200'; // Very Light Orange
  };

  return (
    <div className="flex-1 bg-gray-900 p-2 py-2 pb-24">
      <div className="max-w-4xl w-full mx-auto">
        {/* Model Selection Buttons */}
        <div className="text-orange-500 text-lg font-bold py-2">Algorithms used:</div>
        <div className="flex justify-around mb-4">
          {Object.keys(confMatrixData).map((modelKey) => (
            <button
              key={modelKey}
              onClick={() => setSelectedModel(modelKey)}
              className={`p-2 border ${selectedModel === modelKey ? 'border-orange-500' : 'border-gray-500'} rounded-lg`}
            >
              <span className={`${selectedModel === modelKey ? 'font-bold' : ''} text-white`}>
                {modelKey.replace(/([A-Z])/g, ' $1')}
              </span>
            </button>
          ))}
        </div>

        {/* Confusion Matrix Display */}
        <div>
          {/* Header Information */}
          <div className="p-4 border border-orange-500 rounded-lg mb-4 bg-gray-800">
            <p className="text-white text-sm">
              The predictive model displayed is currently set to{' '}
              <span className="font-bold">{selectedModel.replace(/([A-Z])/g, ' $1')}</span> with an accuracy of {modelData.accuracy}%.
              <br /><br />
              This model predicted {totalCorrectPredictions} correct predictions out of {totalPredictions} total predictions.
              <br /><br />
              Below is the breakdown of the model's predictions, showing how well it distinguishes between each category.
              <br /><br />
              Each row represents the actual class, while each column represents the predicted class.
            </p>
          </div>

          {/* Column Labels */}
          <div className="flex flex-row justify-between items-center border-b border-gray-600 p-2">
            <div className="w-1/5"></div>
            <div className="w-1/5 text-center text-white font-bold">Caution</div>
            <div className="w-1/5 text-center text-white font-bold">Extreme Caution</div>
            <div className="w-1/5 text-center text-white font-bold">Danger</div>
            <div className="w-1/5 text-center text-white font-bold">Extreme Danger</div>
          </div>

          {/* Matrix Rows */}
          {modelData.data.map((row, rowIndex) => (
            <div key={rowIndex} className="flex flex-row justify-between items-center border-b border-gray-600 py-2">
              <div className="w-1/5 text-white font-bold">{row.trueLabel}</div>
              {row.predictedLabels.map((value, index) => (
                <div
                  key={index}
                  className={`w-1/5 p-2 text-center ${getStyleForValue(value)}`}
                >
                  <span className="text-black">{value}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ConfusionMatrix;
