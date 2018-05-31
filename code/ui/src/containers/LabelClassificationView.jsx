import { connect } from 'react-redux';
import React from 'react';
import { submitJudgement } from '../actions';

export const CLASSIFICATION_COLORS = [
"#2ecc71",
"#9b59b6",
"#f1c40f",
"#e74c3c",
"#16a085",
"#27ae60",
"#2980b9",
"#8e44ad",
"#2c3e50",
"#f39c12",
"#d35400",
"#c0392b",
"#1abc9c",
"#3498db",
"#34495e",
"#e67e22",
]

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
