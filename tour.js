export default function({data, parentElement}) {
  const zh = data === "中文";
  parentElement.querySelector("#quant-tour-open").textContent = zh ? "操作导览" : "Guided tour";
  const label = (en, cn) => zh ? cn : en;
  const storeKey = "shanhai-quant-guided-tour-2";
  const old = window.quantResearchGuide;
  let step = old?.activeStep() ?? -1;
  old?.dispose();
  const events = new AbortController();
  const options = {signal: events.signal};
  const steps = [
    {tab:0, selector:".st-key-backtest_asset", title:label("Choose what to backtest","选择回测对象"), copy:label("Choose Stocks to test a stock, Bond ETFs to test a bond fund, or Stock–bond portfolio to compare an allocation of both.","选择「股票」测试单只股票，「债券 ETF」测试单只债券基金，或选择「股债组合」比较两类资产的组合表现。")},
    {selector:".st-key-strategy", sidebar:true, title:label("Choose a trading rule","选择策略"), copy:label("This signal sets entry and exit rules for a single stock or bond ETF. A stock–bond portfolio uses its allocation rule in the main panel.","这里的信号决定单只股票或债券 ETF 的买卖时点。股债组合的配置规则在主面板中选择。")},
    {tab:0, selector:".st-key-run_backtest button, .st-key-run_balanced button", title:label("Run the backtest","运行回测"), copy:label("Run the selected setup here. For a stock or bond ETF, review trades and research reports. For a stock–bond portfolio, compare the allocation with its 60/40 benchmark.","点击这里运行当前方案。单只股票或债券 ETF 可查看交易和研报；股债组合可比较自身表现与 60/40 基准。")},
    {tab:5, selector:".st-key-weight_method", title:label("Compare allocations","比较组合方案"), copy:label("Choose Core–satellite or another weighting rule and run the portfolio. Compare their net performance over the final 30% of the sample.","选择「核心–卫星」等权重方案并运行组合回测，在最后 30% 的样本上比较各方案扣费后的表现。")},
    {tab:6, selector:".st-key-run_balanced button", title:label("Backtest a stock–bond mix","回测股债组合"), copy:label("Choose a US or Chinese ETF pair above. Compare fixed weights or inverse volatility with 60/40, then change the stock and yield shocks below the results.","在上方选择美股或中国 ETF 组合，将固定配比或逆波动率与 60/40 基准对比，再调整结果下方的股价和利率冲击。")},
    {tab:8, selector:".st-key-run_outlook button", title:label("Build a stock outlook","生成个股预测"), copy:label("Edit the stock list and forecast horizon above this button. Run it to see the ranking, error bands and walk-forward evidence.","在按钮上方调整股票池和预测期限。点击后可查看排名、误差区间和历史样本外检验。")},
    {tab:7, title:label("Read the research note","阅读研报"), copy:label("After a single-asset backtest, generate a research note here. The local version needs no API key. Download the note and its evidence together.","单标的回测完成后，在这里生成研究报告。本地版本无需 API Key，正文和证据包都可下载。")}
  ];
  let panel, anchor, timer, pending, autoOpenedSidebar = false;
  const visible = (el) => el && el.getClientRects().length && el.getBoundingClientRect().right>0;
  const tabs = () => [...document.querySelectorAll('[data-testid="stTabs"] [role="tablist"]')]
    .find(visible)?.querySelectorAll('[role="tab"]');
  function scopeFor(tab) {
    if (!tab || tab.getAttribute("aria-selected") !== "true") return null;
    const root = tab.closest('[data-testid="stTabs"]');
    const panelId = tab.getAttribute("aria-controls");
    return [...root.querySelectorAll('[role="tabpanel"]')].find(el =>
      visible(el) && (el.id === panelId || (tab.id && el.getAttribute("aria-labelledby") === tab.id))
    );
  }
  function clearPanel() {
    clearTimeout(timer); cancelAnimationFrame(pending);
    anchor?.classList.remove("quant-tour-anchor"); anchor=null;
    if(panel?.matches(":popover-open")) panel.hidePopover();
    panel?.remove(); panel=null;
  }
  function finish() {
    clearPanel(); step=-1;
    if(autoOpenedSidebar) {
      document.querySelector('[data-testid="stSidebarCollapseButton"] button,button[data-testid="stSidebarCollapseButton"]')?.click();
      autoOpenedSidebar=false;
    }
    try {localStorage.setItem(storeKey,"seen")} catch {}
    document.querySelector("#quant-tour-open")?.focus({preventScroll:true});
  }
  function position() {
    if (!panel || !anchor?.isConnected) return;
    const rect=anchor.getBoundingClientRect(), w=panel.offsetWidth, h=panel.offsetHeight;
    const left=Math.max(10,Math.min(innerWidth-w-10,rect.left));
    const below=rect.bottom+10, above=rect.top-h-10;
    const top=below+h<=innerHeight-8?below:above>=8?above:Math.max(8,Math.min(innerHeight-h-8,below));
    panel.style.left=left+"px"; panel.style.top=top+"px";
  }
  function paint(attempt=0) {
    if(step<0) return;
    const item=steps[step], tab=item.tab===undefined?null:tabs()?.[item.tab];
    if(tab && tab.getAttribute("aria-selected")!=="true") tab.click();
    // A Streamlit rerun can leave a hidden copy of the previous form in the DOM.
    // Resolve every tab target within its own selected, visible panel.
    const scope = item.tab === undefined ? document : scopeFor(tab);
    const target = item.selector
      ? scope && [...scope.querySelectorAll(item.selector)].find(visible)
      : tab?.getAttribute("aria-selected") === "true" && tab;
    if(!visible(target) && item.sidebar && !autoOpenedSidebar) {
      const toggle=document.querySelector('[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapsedControl"] button');
      if(visible(toggle)) {toggle.click();autoOpenedSidebar=true;}
    }
    if(!visible(target)) {
      timer=setTimeout(()=>paint(attempt+1),attempt<30?100:400);
      return;
    }
    anchor=target;
    target.scrollIntoView({block:"center",inline:"nearest",behavior:"instant"});
    target.classList.add("quant-tour-anchor");
    panel=document.createElement("aside"); panel.id="quant-tour"; panel.setAttribute("role","dialog");
    panel.setAttribute("aria-labelledby","quant-tour-title"); panel.setAttribute("aria-live","polite");
    const append = (parent, tag, text) => {
      const node=document.createElement(tag);
      if(text) node.textContent=text;
      parent.append(node); return node;
    };
    const button=(parent,action,text)=>{
      const node=append(parent,"button",text);
      node.type="button";node.dataset.action=action;return node;
    };
    const head=append(panel,"header");
    append(head,"strong",item.title).id="quant-tour-title";
    button(head,"skip",label("Skip","跳过"));
    append(panel,"p",item.copy);
    const foot=append(panel,"footer");
    append(foot,"small",(step+1)+" / "+steps.length);
    button(foot,"back",label("Back","上一步")).disabled=step===0;
    const next=button(foot,"next",step===steps.length-1?label("Finish","完成"):label("Next","下一步"));
    panel.setAttribute("popover", "manual");
    (document.querySelector('[data-testid="stApp"]') || document.body).append(panel);
    panel.showPopover();
    position();
    next.focus({preventScroll:true});
  }
  function go(n) {
    clearPanel();
    if(!steps[n]?.sidebar && (autoOpenedSidebar || (innerWidth<768 && document.querySelector('[data-testid="stSidebar"][aria-expanded="true"]')))) {
      document.querySelector('[data-testid="stSidebarCollapseButton"] button,button[data-testid="stSidebarCollapseButton"]')?.click();
      autoOpenedSidebar=false;
    }
    if(n>=steps.length) {finish();return;}
    step=Math.max(0,n);paint();
  }
  document.addEventListener("click",e=>{
    if(e.target.closest("#quant-tour-open")) {go(0);return;}
    const action=e.target.closest("#quant-tour [data-action]")?.dataset.action;
    if(action==="skip") finish();
    if(action==="back") go(step-1);
    if(action==="next") go(step+1);
  },options);
  document.addEventListener("keydown",e=>{if(step>=0 && e.key==="Escape") {finish();e.preventDefault();}},options);
  function schedule() {cancelAnimationFrame(pending);pending=requestAnimationFrame(position);}
  document.addEventListener("scroll",schedule,{capture:true,signal:events.signal});
  window.addEventListener("resize",schedule,options);
  // Keyed widgets can be replaced after their tab or asset mode changes.
  const observer = new MutationObserver(() => {
    if (step < 0 || !panel || (anchor?.isConnected && visible(anchor))) return;
    clearPanel();
    timer = setTimeout(paint, 80);
  });
  observer.observe(document.body, {childList:true, subtree:true});
  function dispose() {
    events.abort(); observer.disconnect(); clearPanel();
  }
  window.quantResearchGuide={activeStep:()=>step,dispose};
  let seen=false;try {seen=!!localStorage.getItem(storeKey)}catch{}
  if(step>=0 || !seen) {const initial=step<0?0:step;timer=setTimeout(()=>go(initial),350);}
  return dispose;
}
