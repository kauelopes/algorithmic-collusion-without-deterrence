// Abas e idioma. Estado na URL: #aba ou #aba-pt (ex.: #auditorias-pt).
(function () {
  var ABAS = ["visao", "metodo", "auditorias", "incentivos", "paper"];
  var html = document.documentElement;

  function setLang(lang) {
    html.setAttribute("lang", lang);
    document.title = lang === "pt-BR"
      ? "Colusão Algorítmica sem Dissuasão"
      : "Algorithmic Collusion Without Deterrence";
    document.querySelectorAll("#seletores .segmentado button").forEach(function (b) {
      b.classList.toggle("ativa", b.getAttribute("data-lang") === lang);
    });
    try { localStorage.setItem("lang", lang); } catch (e) {}
  }

  function setAba(aba) {
    if (ABAS.indexOf(aba) < 0) aba = ABAS[0];
    document.querySelectorAll("#abas button").forEach(function (b) {
      b.classList.toggle("ativa", b.getAttribute("data-aba") === aba);
    });
    document.querySelectorAll("#palco .aba").forEach(function (s) {
      s.classList.toggle("ativa", s.getAttribute("data-aba") === aba);
    });
    document.querySelectorAll("#painel .painel-aba").forEach(function (p) {
      p.hidden = p.getAttribute("data-aba") !== aba;
    });
    document.getElementById("palco").scrollTop = 0;
  }

  function writeHash(aba, lang) {
    var h = aba + (lang === "pt-BR" ? "-pt" : "");
    if (history.replaceState) history.replaceState(null, "", "#" + h);
  }

  function readHash() {
    var h = (location.hash || "").replace("#", "");
    var lang = null, aba = null;
    if (h) {
      if (/-pt$/.test(h)) { lang = "pt-BR"; h = h.replace(/-pt$/, ""); }
      else if (/-en$/.test(h)) { lang = "en"; h = h.replace(/-en$/, ""); }
      if (h === "pt") { lang = "pt-BR"; h = ""; }
      if (h === "en") { lang = "en"; h = ""; }
      if (ABAS.indexOf(h) >= 0) aba = h;
    }
    return { aba: aba, lang: lang };
  }

  var init = readHash();
  var lang = init.lang;
  if (!lang) { try { lang = localStorage.getItem("lang"); } catch (e) {} }
  if (lang !== "pt-BR" && lang !== "en") lang = "en";
  var aba = init.aba || ABAS[0];
  setLang(lang); setAba(aba); writeHash(aba, lang);

  document.querySelectorAll("#abas button").forEach(function (b) {
    b.addEventListener("click", function () {
      aba = b.getAttribute("data-aba"); setAba(aba); writeHash(aba, lang);
    });
  });
  document.querySelectorAll("#seletores .segmentado button").forEach(function (b) {
    b.addEventListener("click", function () {
      lang = b.getAttribute("data-lang"); setLang(lang); writeHash(aba, lang);
    });
  });
  window.addEventListener("hashchange", function () {
    var s = readHash();
    if (s.lang) { lang = s.lang; setLang(lang); }
    if (s.aba) { aba = s.aba; setAba(aba); }
  });
})();
