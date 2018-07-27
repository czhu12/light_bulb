import { connect } from 'react-redux';
import React from 'react';
import DonePage from './DonePage';
import NavigationBar from './NavigationBar';
import TextTaskView from './TextTaskView';
import ImageTaskView from './ImageTaskView';
import ImageTaskBatchView from './ImageTaskBatchView';
import TaskDescriptionView from './TaskDescriptionView';
import BoundingBoxImageTaskView from './BoundingBoxImageTaskView';
import Footer from './Footer';

class LabelApp extends React.Component {
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
      }
    }

    let style = this.props.done ? { display: "none" } : {}
    let footer = null;

    if (!this.props.task.isBatchView) {
      footer = (<Footer task={this.props.task} currentPrediction={currentPrediction} />);
    }

    return (
      <div>
        <DonePage done={this.props.done}/>
        <div id="wrap" style={style}>
					<NavigationBar />
          <TaskDescriptionView
            title={this.props.task.title}
            description={this.props.task.description}
          />
          <div className="center">
            {taskView}
          </div>
        </div>
        { footer }
      </div>
    );
  }
}

const mapStateToProps = state => ({
  task: state.task,
  items: state.items.items,
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
