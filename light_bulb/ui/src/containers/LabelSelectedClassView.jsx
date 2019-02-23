import { connect } from 'react-redux';
import React from 'react';
import { setCurrentSelectedClass } from '../actions';
import { CLASSIFICATION_COLORS } from '../constants';

class LabelSelectedClassView extends React.Component {
  render() {
    let classificationView = this.props.task.classes.map((cls, index) => {
      let unselectedStyle = {
        color: CLASSIFICATION_COLORS[index],
        borderBottom: `5px solid ${CLASSIFICATION_COLORS[index]}`,
      };

      let selectedStyle = {
        background: CLASSIFICATION_COLORS[index],
        borderBottom: `5px solid ${CLASSIFICATION_COLORS[index]}`,
      }

      let style = index === this.props.currentSelectedClass ? selectedStyle : unselectedStyle;

      return (
        <div className="flex-grow-1">
          <h3
            style={style}
            data-judgement={cls}
            className="judgement-button center"
            onClick={this.props.onSetCurrentSelectedClass.bind(null, index)}
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
  currentSelectedClass: state.items.currentSelectedClass,
});

const mapDispatchToProps = dispatch => ({
  onSetCurrentSelectedClass: (currentSelectedClass) => {
    dispatch(setCurrentSelectedClass(currentSelectedClass))
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(LabelSelectedClassView);
