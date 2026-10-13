// Language switch, table-of-contents highlighting, and the live map of a converged pair.
(function () {
  var html = document.documentElement;
  var T = {
    en: {
      title: "Algorithmic Collusion Without Deterrence",
      start: "Start: firm 1 at {a}, firm 2 at {b}  (CI {c})",
      step: "t = {t}: ({a}, {b})",
      back: "Back on the attractor after {n} step(s).",
      already: "Already on the attractor: ({a}, {b}), CI {c}.",
      loading: "Loading the pair…",
      fail: "The map needs docs/data/pair.json; open the site over HTTP.",
      f1: "firm 1, last price", f2: "firm 2, last price", ne: "competitive", m: "monopoly"
    },
    "pt-BR": {
      title: "Colusão Algorítmica sem Dissuasão",
      start: "Início: firma 1 em {a}, firma 2 em {b}  (CI {c})",
      step: "t = {t}: ({a}, {b})",
      back: "De volta ao atrator em {n} passo(s).",
      already: "Já no atrator: ({a}, {b}), CI {c}.",
      loading: "Carregando o par…",
      fail: "O mapa precisa de docs/data/pair.json; abra o site por HTTP.",
      f1: "firma 1, último preço", f2: "firma 2, último preço", ne: "competitivo", m: "monopólio"
    }
  };
  var lang = "en";
  var demo = { ready: false };
  function t(k, v) {
    var s = T[lang][k];
    Object.keys(v || {}).forEach(function (key) { s = s.replace("{" + key + "}", v[key]); });
    return s;
  }
  function fmt(x) { return lang === "pt-BR" ? x.toFixed(2).replace(".", ",") : x.toFixed(2); }

  // ---------- idioma ----------
  function setLang(l) {
    lang = l;
    html.setAttribute("lang", l);
    document.title = T[l].title;
    document.querySelectorAll("#barra .idioma button").forEach(function (b) {
      b.classList.toggle("ativa", b.getAttribute("data-lang") === l);
    });
    try { localStorage.setItem("lang", l); } catch (e) {}
    if (demo.ready) demo.relabel();
  }
  var fromHash = /(^#|-)pt$/.test(location.hash) ? "pt-BR" : (/(^#|-)en$/.test(location.hash) ? "en" : null);
  var saved = null; try { saved = localStorage.getItem("lang"); } catch (e) {}
  setLang(fromHash || (saved === "pt-BR" ? "pt-BR" : "en"));
  document.querySelectorAll("#barra .idioma button").forEach(function (b) {
    b.addEventListener("click", function () { setLang(b.getAttribute("data-lang")); });
  });
  // #secao-pt rola para a seção e muda o idioma
  var secHash = location.hash.replace(/^#/, "").replace(/-(pt|en)$/, "");
  if (secHash && document.getElementById(secHash)) {
    setTimeout(function () { document.getElementById(secHash).scrollIntoView(); }, 50);
  }

  // ---------- sumário ----------
  var links = Array.prototype.slice.call(document.querySelectorAll("#sumario a"));
  var secs = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
  function spy() {
    var y = window.scrollY + 120, atual = 0;
    secs.forEach(function (s, i) { if (s && s.offsetTop <= y) atual = i; });
    links.forEach(function (a, i) { a.classList.toggle("ativa", i === atual); });
  }
  window.addEventListener("scroll", spy, { passive: true });
  spy();

  // ---------- mapa ao vivo ----------
  var svg = document.getElementById("grade");
  var leitura = document.getElementById("leitura");
  if (svg) {
    leitura.textContent = t("loading");
    fetch("./data/pair.json").then(function (r) { return r.json(); }).then(montar).catch(function () {
      leitura.textContent = t("fail");
    });
  }

  function montar(D) {
    var n = D.grid.length, X0 = 46, Y0 = 14, L = 338, c = L / n;
    var NS = "http://www.w3.org/2000/svg";
    var att = {}; D.attractor.forEach(function (s) { att[s[0] + "," + s[1]] = true; });
    var reduzido = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var timer = null, cells = {};

    function el(tag, attrs, parent) {
      var e = document.createElementNS(NS, tag);
      Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
      (parent || svg).appendChild(e); return e;
    }
    function cx(i) { return X0 + (i + 0.5) * c; }       // firma 1 no eixo x (esquerda = barato)
    function cy(j) { return Y0 + (n - 1 - j + 0.5) * c; } // firma 2 no eixo y (baixo = barato)
    function pos(p) { // posição fracionária de um preço na grade
      var k = 0; while (k < n - 1 && D.grid[k + 1] <= p) k++;
      var f = (p - D.grid[k]) / (D.grid[k + 1] - D.grid[k]); return k + f;
    }
    function ci(p) { return (p - D.p_ne) / (D.p_m - D.p_ne); }

    svg.innerHTML = "";
    var gCells = el("g", {});
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var r = el("rect", { x: X0 + i * c, y: Y0 + (n - 1 - j) * c, width: c, height: c, "class": "cel" }, gCells);
      r.dataset.i = i; r.dataset.j = j;
      if (att[i + "," + j]) r.classList.add("att");
      cells[i + "," + j] = r;
      r.addEventListener("click", (function (ii, jj) { return function () { rodar(ii, jj); }; })(i, j));
    }
    // linhas de referência (competitivo e monopólio)
    var gRef = el("g", {});
    [["ne", D.p_ne], ["m", D.p_m]].forEach(function (ref) {
      var q = pos(ref[1]) + 0.5;
      el("line", { x1: X0 + q * c, y1: Y0, x2: X0 + q * c, y2: Y0 + L, "class": "ref" }, gRef);
      el("line", { x1: X0, y1: Y0 + (n - q) * c, x2: X0 + L, y2: Y0 + (n - q) * c, "class": "ref" }, gRef);
      var tx = el("text", { x: X0 + q * c + 3, y: Y0 + 9, "class": "ref-txt", "data-ref": ref[0] }, gRef);
      tx.textContent = t(ref[0]);
    });
    // eixos
    var gAx = el("g", {});
    for (var k = 0; k < n; k += 2) {
      var tx1 = el("text", { x: cx(k), y: Y0 + L + 11, "text-anchor": "middle", "class": "eixo" }, gAx); tx1.textContent = D.grid[k].toFixed(2);
      var tx2 = el("text", { x: X0 - 4, y: cy(k) + 3, "text-anchor": "end", "class": "eixo" }, gAx); tx2.textContent = D.grid[k].toFixed(2);
    }
    var ax1 = el("text", { x: X0 + L / 2, y: Y0 + L + 26, "text-anchor": "middle", "class": "eixo-titulo", "data-ax": "f1" }, gAx);
    var ax2 = el("text", { x: 10, y: Y0 + L / 2, "class": "eixo-titulo", "data-ax": "f2", transform: "rotate(-90 10 " + (Y0 + L / 2) + ")", "text-anchor": "middle" }, gAx);
    var gPath = el("g", {});
    var path = el("polyline", { "class": "traj", points: "" }, gPath);

    function relabel() {
      ax1.textContent = t("f1"); ax2.textContent = t("f2");
      gRef.querySelectorAll("text").forEach(function (x) { x.textContent = t(x.getAttribute("data-ref")); });
      if (!timer) mostrarAtrator();
    }
    function mostrarAtrator() {
      var s = D.attractor[0];
      leitura.textContent = t("already", { a: fmt(D.grid[s[0]]), b: fmt(D.grid[s[1]]), c: fmt(ci(D.grid[s[0]])) });
    }
    function limpar() {
      if (timer) { clearInterval(timer); timer = null; }
      Object.keys(cells).forEach(function (k) { cells[k].classList.remove("rota", "atual"); });
      path.setAttribute("points", "");
    }
    function rodar(i, j) {
      limpar();
      var traj = [[i, j]], seen = {}; seen[i + "," + j] = true;
      var ii = i, jj = j, guard = 0;
      while (!att[ii + "," + jj] && guard < 400) {
        var ni = D.next0[ii][jj], nj = D.next1[jj][ii];
        ii = ni; jj = nj; traj.push([ii, jj]); guard++;
        if (seen[ii + "," + jj]) break; seen[ii + "," + jj] = true;
      }
      var k = 0, pts = [];
      var linhas = [t("start", { a: fmt(D.grid[i]), b: fmt(D.grid[j]), c: fmt(ci(D.grid[i])) })];
      function passo() {
        var s = traj[k];
        Object.keys(cells).forEach(function (q) { cells[q].classList.remove("atual"); });
        cells[s[0] + "," + s[1]].classList.add("atual");
        if (k > 0) cells[traj[k - 1][0] + "," + traj[k - 1][1]].classList.add("rota");
        pts.push(cx(s[0]) + "," + cy(s[1])); path.setAttribute("points", pts.join(" "));
        if (k > 0) linhas.push(t("step", { t: k, a: fmt(D.grid[s[0]]), b: fmt(D.grid[s[1]]) }));
        if (k === traj.length - 1) {
          if (k === 0) linhas = [t("already", { a: fmt(D.grid[s[0]]), b: fmt(D.grid[s[1]]), c: fmt(ci(D.grid[s[0]])) })];
          else linhas.push(t("back", { n: k }));
          leitura.textContent = linhas.slice(-6).join("\n");
          clearInterval(timer); timer = null; return;
        }
        leitura.textContent = linhas.slice(-6).join("\n");
        k++;
      }
      if (reduzido) {
        // sem animação: percorre a trajetória inteira de uma vez
        var fim = false;
        while (!fim) { fim = (k === traj.length - 1); passo(); }
      } else { passo(); timer = setInterval(passo, 420); }
    }
    var a0 = D.attractor[0];
    document.getElementById("b-corte").addEventListener("click", function () { rodar(0, a0[1]); });
    document.getElementById("b-alta").addEventListener("click", function () { rodar(n - 1, a0[1]); });
    document.getElementById("b-rival").addEventListener("click", function () { rodar(a0[0], 0); });
    document.getElementById("b-aleatorio").addEventListener("click", function () {
      rodar(Math.floor(Math.random() * n), Math.floor(Math.random() * n));
    });
    demo.ready = true; demo.relabel = relabel; relabel();
  }
})();
