import { connect } from 'react-redux';
import React from 'react';
import { CLASSIFICATION_COLORS } from '../constants';
import { submitJudgement } from '../actions';

function isJsonString(str) {
    try {
      JSON.parse(str);
    } catch (e) {
      return false;
    }
    return true;
}


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
    let sequence = JSON.parse(props.currentItem['text']);
    // If there is already a prediction
    if (isJsonString(props.currentItem['label'])) {
      let labels = JSON.parse(props.currentItem['label']);
      let classes = props.task.classes;
      return labels.map((label) => {
        let word = label['word'];
        let tag = label['tag'];
        let tagIdx = classes.indexOf(tag);
        tagIdx = tagIdx === -1 ? null : tagIdx;
        return {
          word: word,
          classIdx: tagIdx,
        };
      })
    } else {
      let labels = sequence.map((word, index) => {
        return {word, classIdx: null}
      })
      return labels
    }
  }

  clickedWord(e) {
    let index = parseInt(e.target.getAttribute('data-word-index'), 10);
    let labels = this.state.labels.slice();
    if (labels[index]['classIdx'] === this.props.currentSelectedClass) {
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

    if (this.props.isBatchMode) {
      this.props.changedLabels(labels.map((l) => {
        let tag = l['classIdx'] !== null ? this.props.task.classes[l['classIdx']] : this.props.task.defaultClass;
        return {
          tag: tag,
          word: l['word'],
        }
      }));
    }
    this.setState({labels});
  }

	_submitLabels(e) {
    let key = e.which || e.keyCode;
    if (key === 13) {
      this.props.submitLabels(this.state.labels, this.props.task);
    }
	}

	componentDidMount() {
    if (!this.props.isBatchMode) {
      window.addEventListener("keypress", this._submitLabels.bind(this));
    }
	}

	componentWillUnmount() {
    if (!this.props.isBatchMode) {
      window.removeEventListener("keypress", this._submitLabels.bind(this));
    }
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
    
    dispatch(submitJudgement(labelsToSubmit));
  },
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SequenceTaggerTaskView);
