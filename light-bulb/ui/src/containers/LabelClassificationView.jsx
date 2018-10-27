import { connect } from 'react-redux';
import React from 'react';
import { CLASSIFICATION_COLORS } from '../constants';
import { submitJudgement } from '../actions';

class LabelClassificationView extends React.Component {
  render() {
    let classificationView = this.props.task.classes.map((cls, index) => {
      let style = { background: CLASSIFICATION_COLORS[index] }
      return (
        <div className="flex-grow-1">
          <h3
            style={style}
            data-judgement={cls}
            className="judgement-button center"
            onClick={this.props.onJudgement.bind(null, cls)}
          >
            {cls}
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
