import { connect } from 'react-redux';
import React from 'react';
import { CLASSIFICATION_COLORS } from '../constants';
import { submitJudgement } from '../actions';

class LabelClassificationView extends React.Component {
	componentDidMount() {
    window.addEventListener("keypress", this._submitLabelByKeyboard.bind(this));
	}

	componentWillUnmount() {
    window.removeEventListener("keypress", this._submitLabelByKeyboard.bind(this));
	}

  _submitLabelByKeyboard(e) {
    // Search for this key in the classes;
    let key = e.which || e.keyCode;
    let targetChar = String.fromCharCode(key);
    let foundIndex = this.props.task.classes.map((cls) => {
      return cls[0].toLowerCase();
    }).findIndex((classChar) => {
      return classChar === targetChar;
    });

    if (foundIndex >= 0) {
      let cls = this.props.task.classes[foundIndex];
      this.props.onJudgement(cls);
    }
  }

  _allClassNamesStartWithDifferentLetters(classes) {
    let map = {};
    for (let i = 0; i < classes.length; i++) {
      if (classes[i][0] in map) return false;
      map[classes[i][0]] = true;
    }
    return true;
  }

  render() {
    let classificationView = this.props.task.classes.map((cls, index) => {
      let style = { background: CLASSIFICATION_COLORS[index] }
      let classNameText = null;
      if (this._allClassNamesStartWithDifferentLetters(this.props.task.classes)) {
        classNameText = (<div><u>{cls[0]}</u>{cls.substring(1, cls.length)}</div>);
      } else {
        classNameText = cls;
      }
      return (
        <div className="flex-grow-1">
          <h3
            style={style}
            data-judgement={cls}
            className="judgement-button center"
            onClick={this.props.onJudgement.bind(null, cls)}
          >
            {classNameText}
          </h3>
        </div>
      );
    });

    return (
      <div className="flex">
        {classificationView}
      </div>
    );
  }
}

const mapStateToProps = state => ({
});


const mapDispatchToProps = dispatch => ({
  onJudgement: (cls) => {
    dispatch(submitJudgement(cls))
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(LabelClassificationView);
