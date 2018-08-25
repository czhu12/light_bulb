import { connect } from 'react-redux';
import React from 'react';
import { setBoundingBoxClass } from '../actions';
import { CLASSIFICATION_COLORS } from '../constants';

class LabelBoundingBoxClassView extends React.Component {
  render() {
    let classificationView = this.props.task.classes.map((cls, index) => {
      let style = { background: CLASSIFICATION_COLORS[index] }
      return (
        <div className="flex-grow-1">
          <h3
            style={style}
            data-judgement={cls}
            className="judgement-button center"
            onClick={this.props.onSetBoundingBoxClass.bind(null, cls)}
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
  currentBoundingBoxClass: state.items.currentBoundingBoxClass,
});


const mapDispatchToProps = dispatch => ({
  onSetBoundingBoxClass: (boundingBoxClass) => {
    dispatch(setBoundingBoxClass(boundingBoxClass))
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(LabelBoundingBoxClassView);
