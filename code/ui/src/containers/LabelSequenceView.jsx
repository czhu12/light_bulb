import React from 'react';

class LabelSequenceView extends React.Component {
  render() {
    let validTokensView = this.props.task.validTokens.map((token) => {
      return (<div className="token center" data-token={token}>{token}</div>)
    });
    let taskTitle = this.props.task.title;
    let suggestion = "B-LOC I-LOC O I-VERT"
    return (
      <div>
        <div id="control-panel">
          <div>
            Suggestion: <span id="suggestion">{suggestion}</span>
          </div>

          <div>
            {validTokensView}
            <div className="clear"></div>
          </div>
        </div>

        <div className="sequence-input">
          <div className="input-group mb-3">
            <div className="input-group-prepend">
              <span className="input-group-text">{taskTitle}</span>
            </div>

            <input
              autofocus
              type="text"
              className="form-control"
              placeholder="label" />
          </div>
        </div>
      </div>
    );
  }
}

export default LabelSequenceView;
