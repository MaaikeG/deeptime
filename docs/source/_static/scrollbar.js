const sideContainer = document.getElementsByClassName('sphinxsidebar');

const ps = new PerfectScrollbar(".sphinxsidebar", {
    wheelPropagation: true
});
ps.update()
