import { connect } from 'react-redux';
import React from 'react';

import LabelSequenceView from './LabelSequenceView';
import LabelClassificationView from './LabelClassificationView';

class Footer extends React.Component {
  render() {
    let labelView = null;
    if (this.props.task.labelType === 'classification') {
      labelView = (
        <LabelClassificationView task={this.props.task} />
      );
    } else if (this.props.task.labelType === 'binary') {
      labelView = (
        <LabelClassificationView task={this.props.task} />
      );
    } else if (this.props.task.labelType === 'sequence') {
      labelView = (
        <LabelSequenceView task={this.props.task} />
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
  task: state.task,
});


const mapDispatchToProps = dispatch => ({
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(Footer);
