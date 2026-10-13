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
      campo: "From ({a}, {b}): mean next state ({c}, {d}) over {n} pairs.\nMean step length {m}; on the attractor of {s}% of the pairs.",
      campoDica: "Click a cell to read the mean step from it.",
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
      campo: "De ({a}, {b}): próximo estado médio ({c}, {d}) em {n} pares.\nPasso médio {m}; no atrator de {s}% dos pares.",
      campoDica: "Clique numa célula para ler o passo médio a partir dela.",
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
  // Dois modos: "par" anima o mapa determinístico de um par (pair.json);
  // "campo" desenha o deslocamento médio por célula numa população (population.json).
  var svg = document.getElementById("grade");
  var leitura = document.getElementById("leitura");
  var demoEl = document.getElementById("demo");
  var dados = { par: null, campo: null }, modo = "par";
  if (svg) {
    leitura.textContent = t("loading");
    fetch("./data/pair.json").then(function (r) { return r.json(); }).then(function (D) {
      dados.par = D; montar();
    }).catch(function () { leitura.textContent = t("fail"); });
    fetch("./data/population.json").then(function (r) { return r.json(); }).then(function (P) {
      dados.campo = P;
      document.querySelectorAll("#demo .seletor button").forEach(function (b) { b.disabled = false; });
      if (modo === "campo") montar();
    }).catch(function () {});
    document.querySelectorAll("#demo .seletor button").forEach(function (b) {
      b.addEventListener("click", function () {
        var m = b.getAttribute("data-modo");
        if (m === modo || !dados[m]) return;
        modo = m; montar();
      });
    });
  }

  function montar() {
    var D = dados[modo];
    if (!D) return;
    var n = D.grid.length, X0 = 46, Y0 = 14, L = 338, c = L / n;
    var NS = "http://www.w3.org/2000/svg";
    var reduzido = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var timer = null, cells = {};
    var att = {};
    if (modo === "par") D.attractor.forEach(function (s) { att[s[0] + "," + s[1]] = true; });
    demoEl.setAttribute("data-modo", modo);
    document.querySelectorAll("#demo .seletor button").forEach(function (b) {
      b.classList.toggle("ativa", b.getAttribute("data-modo") === modo);
      b.setAttribute("aria-pressed", b.getAttribute("data-modo") === modo ? "true" : "false");
    });

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
    var attMax = 0;
    if (modo === "campo") D.att.forEach(function (row) { row.forEach(function (v) { if (v > attMax) attMax = v; }); });
    var gCells = el("g", {});
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var r = el("rect", { x: X0 + i * c, y: Y0 + (n - 1 - j) * c, width: c, height: c, "class": "cel" }, gCells);
      r.dataset.i = i; r.dataset.j = j;
      if (att[i + "," + j]) r.classList.add("att");
      if (modo === "campo" && attMax > 0) {
        r.classList.add("sombra");
        r.style.fillOpacity = (0.08 + 0.92 * D.att[i][j] / attMax).toFixed(3);
      }
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
    // campo vetorial médio: uma seta por célula, comprimento em unidades de célula (truncado)
    var gSetas = el("g", { "class": "setas", "pointer-events": "none" });
    if (modo === "campo") {
      var passo = D.grid[1] - D.grid[0], maxLen = 1.6 * c;
      var defs = el("defs", {});
      var mk = el("marker", { id: "ponta", viewBox: "0 0 6 6", refX: "5", refY: "3", markerWidth: "4", markerHeight: "4", orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L6,3 L0,6 Z", "class": "ponta" }, mk);
      for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) {
        var vx = D.dx[a][b] / passo * c, vy = -D.dy[a][b] / passo * c;
        var len = Math.sqrt(vx * vx + vy * vy);
        if (len < 0.08 * c) { el("circle", { cx: cx(a), cy: cy(b), r: 1.2, "class": "parado" }, gSetas); continue; }
        var f = Math.min(1, maxLen / len);
        el("line", { x1: cx(a), y1: cy(b), x2: cx(a) + vx * f, y2: cy(b) + vy * f, "class": "seta", "marker-end": "url(#ponta)" }, gSetas);
      }
    }
    var gPath = el("g", {});
    var path = el("polyline", { "class": "traj", points: "" }, gPath);

    function relabel() {
      ax1.textContent = t("f1"); ax2.textContent = t("f2");
      gRef.querySelectorAll("text").forEach(function (x) { x.textContent = t(x.getAttribute("data-ref")); });
      if (!timer) { if (modo === "par") mostrarAtrator(); else leitura.textContent = t("campoDica"); }
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
    function lerCampo(i, j) {
      limpar();
      cells[i + "," + j].classList.add("atual");
      var x1 = D.grid[i] + D.dx[i][j], y1 = D.grid[j] + D.dy[i][j];
      path.setAttribute("points", cx(i) + "," + cy(j) + " " + (X0 + (pos(x1) + 0.5) * c) + "," + (Y0 + (n - pos(y1) - 0.5) * c));
      leitura.textContent = t("campo", {
        a: fmt(D.grid[i]), b: fmt(D.grid[j]), c: fmt(x1), d: fmt(y1),
        n: D.n_pairs.toLocaleString(lang === "pt-BR" ? "pt-BR" : "en-US"),
        m: fmt(D.mag[i][j]), s: (100 * D.att[i][j]).toFixed(1).replace(".", lang === "pt-BR" ? "," : ".")
      });
    }
    function rodar(i, j) {
      if (modo === "campo") return lerCampo(i, j);
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
    // os botões partem do atrator do par; no campo, da crista mais frequente
    var a0;
    if (modo === "par") a0 = D.attractor[0];
    else {
      var best = 0, bi = 0, bj = 0;
      for (var u = 0; u < n; u++) for (var v = 0; v < n; v++) if (D.att[u][v] > best) { best = D.att[u][v]; bi = u; bj = v; }
      a0 = [bi, bj];
    }
    demo.botoes = {
      "b-corte": function () { rodar(0, a0[1]); },
      "b-alta": function () { rodar(n - 1, a0[1]); },
      "b-rival": function () { rodar(a0[0], 0); },
      "b-aleatorio": function () { rodar(Math.floor(Math.random() * n), Math.floor(Math.random() * n)); }
    };
    demo.ready = true; demo.relabel = relabel; relabel();
  }
  ["b-corte", "b-alta", "b-rival", "b-aleatorio"].forEach(function (id) {
    var b = document.getElementById(id);
    if (b) b.addEventListener("click", function () { if (demo.botoes) demo.botoes[id](); });
  });
})();
