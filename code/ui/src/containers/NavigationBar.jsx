import { connect } from 'react-redux';
import React from 'react';

class NavigationBar extends React.Component {
  render() {
    let totalItems = this.props.unlabelled + this.props.labelled.total;
    let labelled = this.props.labelled.total;
    let maxAccuracy = 0;
    if (this.props.history.length > 1) {
      maxAccuracy = Math.round(
        Math.max(...this.props.history.map((step) => step['test']['acc'])) * 100
      )
    }

    return (
      <nav className="navbar navbar-toggleable-md navbar-light bg-faded navbar-expand-md">
        <button className="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
          <span className="navbar-toggler-icon"></span>
        </button>
        <a className="navbar-brand" href="#"><i className="far fa-lightbulb"></i></a>
        <div className="collapse navbar-collapse" id="navbarNavAltMarkup">
          <div className="navbar-nav">
            <a className="nav-item nav-link">Labelled: <b id="labelled-counts-text">{labelled} / {totalItems}</b></a>
            <a className="nav-item nav-link">Accuracy: <b id="accuracy-text">{maxAccuracy}%</b></a>
          </div>
        </div>
      </nav>
    );
  }
}

const mapStateToProps = state => ({
  unlabelled: state.stats.unlabelled,
  labelled: state.stats.labelled,
  history: state.stats.history,
  errorMsg: state.judgements.errorMsg || state.stats.errorMsg || state.items.errorMsg,
  fetching: state.judgements.fetching || state.stats.fetching || state.items.fetching,
});


const mapDispatchToProps = dispatch => ({
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(NavigationBar);
