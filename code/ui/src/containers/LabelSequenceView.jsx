import { connect } from 'react-redux';
import React from 'react';
import { changeSequenceInput, submitJudgement } from '../actions';

class LabelSequenceView extends React.Component {
  onTokenClick(token) {
    this.props.onTokenClick(this.props.sequenceInput, token);
    this.refs.sequenceInput.focus();
  }

  render() {
    let validTokensView = this.props.task.validTokens.map((token) => {
      return (
        <div
          className="token center"
          onClick={this.onTokenClick.bind(this, token)}
        >
          {token}
        </div>
      )
    });
    let taskTitle = this.props.task.title;
    let inputClass = this.props.errorMsg ? 'form-control is-invalid' : 'form-control';
    return (
      <div>
        <div id="control-panel">
          <div>
            Suggestion: <span id="suggestion">
            {
              this.props.currentPrediction ?
              this.props.currentPrediction.join(' ') :
              null
            }
            </span>
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
              onChange={this.props.onChangeSequenceInput}
              onKeyPress={this.props.onKeyPress}
              value={this.props.sequenceInput}
              ref="sequenceInput"
              autofocus
              type="text"
              className={inputClass}
              placeholder="label" />
          </div>
          <div class="invalid-feedback" style={{color: 'red'}}>
            <b>{this.props.errorMsg}</b>
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = state => ({
  sequenceInput: state.judgements.sequenceInput,
  errorMsg: state.judgements.errorMsg,
});

const mapDispatchToProps = dispatch => ({
  onChangeSequenceInput: (e) => {
    dispatch(changeSequenceInput(e.target.value));
  },
  onKeyPress: (e) => {
    if (e.key === 'Enter') {
      dispatch(submitJudgement(e.target.value));
      dispatch(changeSequenceInput(''));
    }
  },
  onTokenClick: (sequenceInput, token) => {
    let currentTokens = sequenceInput.split(' ').filter((token) => token);
    currentTokens.push(token);
    dispatch(changeSequenceInput(currentTokens.join(' ').trim()));
  }
});


export default connect(
  mapStateToProps,
  mapDispatchToProps
)(LabelSequenceView);
