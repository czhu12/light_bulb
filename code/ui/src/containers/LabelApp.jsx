import { connect } from 'react-redux';
import React from 'react';

import { CLASSIFICATION_COLORS } from '../constants';
import DonePage from './DonePage';
import NavigationBar from './NavigationBar';
import TextTaskView from './TextTaskView';
import ImageTaskView from './ImageTaskView';
import ImageTaskBatchView from './ImageTaskBatchView';
import TaskDescriptionView from './TaskDescriptionView';
import BoundingBoxImageTaskView from './BoundingBoxImageTaskView';
import SequenceTaggerTaskView from './SequenceTaggerTaskView';
import JSONTaskView from './JSONTaskView';
import Footer from './Footer';

class LabelApp extends React.Component {
  _computeTitle() {
    if (this.props.task.isBatchView) {
      let dataType = this.props.task.dataType;
      let className = this.props.task.classes[this.props.batchItems.targetClass];
      return `Unselect all ${dataType} that is not a <span style="color:${CLASSIFICATION_COLORS[this.props.batchItems.targetClass]}">${className}</span>`;
    } else {
      return this.props.task.title;
    }
  }

  _computeDescription() {
    return this.props.task.isBatchView ? null : this.props.task.description;
  }

  render() {
    let taskView = null;
    let currentPrediction = null;

    if (this.props.currentIndex != null) {
      let currentItem = this.props.items[this.props.currentIndex];
      if (!this.props.predictions && (this.props.currentIndex < this.props.predictions.length)) {
        currentPrediction = this.props.predictions[this.props.currentIndex];
      }

      if (this.props.task.isBatchView) {
        taskView = (
          <ImageTaskBatchView
            items={this.props.items}
            predictions={this.props.predictions}
          />
        );
      } else if (this.props.task.dataType === 'images' && this.props.task.labelType === 'object_detection') {
        taskView = (
          <BoundingBoxImageTaskView
            currentItem={currentItem}
            currentPrediction={currentPrediction}
          />
        );
      } else if (this.props.task.dataType === 'images') {
        taskView = (
          <ImageTaskView
            currentItem={currentItem}
            currentPrediction={currentPrediction}
          />
        );
      } else if (this.props.task.dataType === 'text') {
        taskView = (
          <TextTaskView
            currentItem={currentItem}
            currentPrediction={currentPrediction}
          />
        );
      } else if (this.props.task.dataType === 'json' && this.props.task.labelType === 'sequence') {
        taskView = (
          <SequenceTaggerTaskView
            currentItem={currentItem}
            task={this.props.task}
          />

        );
      } else if (this.props.task.dataType === 'json') {
        taskView = (
          <JSONTaskView
            currentItem={currentItem}
            task={this.props.task}
            currentPrediction={currentPrediction}
          />
        );
      }
    }

    let style = this.props.done ? { display: "none" } : {}
    let footer = null;

    if (!this.props.task.isBatchView) {
      footer = (<Footer task={this.props.task} currentPrediction={currentPrediction} />);
    }

    let fetchingView = this.props.fetching ?
      (<div className="center"><i id="items-loading" className="fa fa-spinner fa-spin" aria-hidden="true"></i></div>) :
        null;

    let taskViewStyle = {
      overflowY: this.props.task.isBatchView ? 'visible' : 'auto',
      maxHeight: this.props.task.isBatchView ? 'none': '400px',
      marginLeft: '20px',
      marginRight: '20px',
    }
    let mainView = null;
    if (this.props.done) {
      return (
        <div>
          <DonePage done={this.props.done}/>
        </div>
      );
    } else {
      return (
        <div>
          <div id="wrap">
            <NavigationBar />
            <TaskDescriptionView
              title={this._computeTitle()}
              description={this._computeDescription()}
            />
            <div
              className="center"
              id="task-view"
              style={taskViewStyle}>
              {taskView}
            </div>
            {fetchingView}
          </div>
          { footer }
        </div>
      );
    }
  }
}

const mapStateToProps = state => ({
  task: state.task,
  items: state.items.items,
  batchItems: state.batchItems,
  predictions: state.items.predictions,
  currentIndex: state.items.currentIndex,
  done: state.items.done,
  fetching: state.items.fetching,
});


const mapDispatchToProps = dispatch => ({
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(LabelApp);
