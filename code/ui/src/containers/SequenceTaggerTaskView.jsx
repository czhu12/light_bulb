import { connect } from 'react-redux';
import React from 'react';
import { CLASSIFICATION_COLORS } from '../constants';
import { changeSequenceInput, submitJudgement } from '../actions';

class SequenceTaggerTaskView extends React.Component {
  constructor(props) {
    super(props);
    let labels = this.initializeTags(props);
    this.state = {labels};
  }

  componentWillReceiveProps(nextProps) {
    if (this.props.currentItem['path'] !== nextProps.currentItem['path']) {
      let labels = this.initializeTags(nextProps);
      this.setState({labels});
    }
  }

  initializeTags(props) {
    let sequence = JSON.parse(this.props.currentItem['text']);
    let labels = sequence.map((word, index) => {
      return {word, classIdx: null}
    })
    return labels
  }

  clickedWord(e) {
    let index = parseInt(e.target.getAttribute('data-word-index'), 10);
    let labels = this.state.labels.slice();
    if (labels[index]['classIdx'] == this.props.currentSelectedClass) {
      // Unset tag
      labels[index] = {
        word: labels[index]['word'],
        classIdx: null,
      }
    } else {
      // Set tag
      labels[index] = {
        word: labels[index]['word'],
        classIdx: this.props.currentSelectedClass,
      }
    }

    this.setState({labels});
  }

	_submitLabels() {
    this.props.submitLabels(this.state.labels, this.props.task);
	}

	componentDidMount() {
		window.addEventListener("click", this._submitLabels);
	}

	componentWillUnmount() {
		window.removeEventListener("click", this._submitLabels);
	}

  render() {
    let sequenceText = this.state.labels.map((label, idx) => {
      let word = label['word'];
      let color = CLASSIFICATION_COLORS[label['classIdx']];

      return (
        <span
          style={{color: color}}
          data-word-index={idx}
          onClick={this.clickedWord.bind(this)}
          className="sequence-word noselect">
          {word}
        </span>
      );
    })
    return (
      <div>
        <div id="text-classification-text">
          {sequenceText}
        </div>
        <div className="clear"></div>
      </div>
    );
  }
}

const mapStateToProps = state => ({
  currentSelectedClass: state.items.currentSelectedClass,
});

const mapDispatchToProps = dispatch => ({
  submitLabels: (labels, task) => {
    let classes = task.classes;
    let labelsToSubmit = labels.map((label) => {
      let classIdx = label['classIdx'];
      let word = label['word'];
      let cls = classes[classIdx];
      // Set class to default class
      if (!cls) cls = task.defaultClass;
      return {word, tag: cls}
    });
    console.log(labelsToSubmit);
    
    dispatch(submitJudgement(labelsToSubmit));
  },
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SequenceTaggerTaskView);
