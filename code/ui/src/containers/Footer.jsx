import { connect } from 'react-redux';
import React from 'react';

import LabelSequenceView from './LabelSequenceView';
import LabelClassificationView from './LabelClassificationView';
import LabelBoundingBoxClassView from './LabelBoundingBoxClassView';

class Footer extends React.Component {
  render() {
    let labelView = null;
    if (
      this.props.task.labelType === 'classification' ||
      this.props.task.labelType === 'binary'
    ) {
      labelView = (
        <LabelClassificationView
          task={this.props.task}
          currentPrediction={this.props.currentPrediction}
        />
      );
    } else if (
      this.props.task.labelType === 'sequence' ||
      this.props.task.labelType === 'seq2seq'
    ) {
      labelView = (
        <LabelSequenceView
          task={this.props.task}
          currentPrediction={this.props.currentPrediction}
        />
      );
    } else if (this.props.task.labelType === 'object_detection') {
      labelView = (
        <LabelBoundingBoxClassView
          task={this.props.task}
          currentPrediction={this.props.currentPrediction}
        />
      );
    }

    return (
      <div id="footer">
        {labelView}
      </div>
    );
  }
}

const mapStateToProps = state => ({
});


const mapDispatchToProps = dispatch => ({
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(Footer);
