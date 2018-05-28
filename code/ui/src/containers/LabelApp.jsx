import { connect } from 'react-redux';
import React from 'react';
import DonePage from './DonePage';
import NavigationBar from './NavigationBar';
import TextTaskView from './TextTaskView';
import ImageTaskView from './ImageTaskView';
import TaskDescriptionView from './TaskDescriptionView';
import Footer from './Footer';

class LabelApp extends React.Component {
  onJudgement(judgement) {
    let currentItem = this.props.items[this.props.currentIndex];
    this.props.onJudgement(currentItem['path'], judgement);
  }

  render() {
    let taskView = null;
    if (this.props.currentIndex != null) {
      let currentItem = this.props.items[this.props.currentIndex];

      if (this.props.task.dataType === 'images') {
        taskView = (
          <ImageTaskView currentItem={currentItem}/>
        );
      } else if (this.props.task.dataType === 'text') {
        taskView = (
          <TextTaskView currentItem={currentItem}/>
        );
      }
    }
    return (
      <div>
        <div id="wrap">
					<NavigationBar />
          <TaskDescriptionView
            title={this.props.task.title}
            description={this.props.task.description}
          />
          <div className="center">
            {taskView}
          </div>
        </div>

        <Footer onJudgement={this.onJudgement} />
      </div>
    );
  }
}

const mapStateToProps = state => ({
  task: state.task,
  items: state.items.items,
  currentIndex: state.items.currentIndex,
  done: state.items.done,
  errorMsg: state.judgements.errorMsg || state.stats.errorMsg || state.items.errorMsg,
  fetching: state.judgements.fetching || state.stats.fetching || state.items.fetching,
});


const mapDispatchToProps = dispatch => ({
  onJudgement: (judgement) => {
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(LabelApp);
