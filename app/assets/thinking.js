/**
 * Instant thinking animation for AIA Agent 360.
 * Runs outside Dash/React via raw DOM event listeners.
 * - Capture phase fires BEFORE Dash processes events
 * - Double-rAF injects AFTER React reconciliation
 * - MutationObserver cleans up when React replaces content
 * - Progressive step animation while waiting
 * - New chat = page reload (server is blocked during agent calls)
 */
(function() {
    "use strict";

    var SAMPLE_QS = [
        "What is the total number of claims by region?",
        "Which product categories have the highest fraud scores?",
        "What does the AIA Health Premium Plan cover?",
        "Are there any anomalies in our claims data?",
        "Show me a dashboard of claims trends by region",
        "Show me the top 5 agents by premium sold"
    ];

    var THINK_STEPS = [
        "Classifying intent...",
        "Resolving assets via Context Index...",
        "Routing to specialist agents...",
        "Querying data sources...",
        "Composing answer..."
    ];

    var _processing = false;
    var _stepInterval = null;

    // Add keyframes
    var style = document.createElement("style");
    style.textContent =
        "@keyframes aiaThinkSpin { to { transform: rotate(360deg); } } " +
        "@keyframes aiaThinkPulse { 0%,100%{opacity:1} 50%{opacity:0.4} } " +
        "@keyframes aiaFadeIn { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }";
    document.head.appendChild(style);

    function makeStepHtml(text, active) {
        var dotColor = active ? "#059669" : "#d1d5db";
        var textColor = active ? "#6b7280" : "#d1d5db";
        var anim = active ? ";animation:aiaFadeIn 0.3s ease-out" : "";
        return '<div style="display:flex;align-items:flex-start;margin-bottom:3px' + anim + '">' +
            '<span style="width:5px;height:5px;border-radius:50%;background:' + dotColor +
            ';display:inline-block;margin-right:6px;margin-top:4px;flex-shrink:0"></span>' +
            '<span style="color:' + textColor + '">' + text + '</span></div>';
    }

    function injectThinking(question) {
        if (!question || _processing) return;
        _processing = true;

        // Clear any previous step animation
        if (_stepInterval) { clearInterval(_stepInterval); _stepInterval = null; }

        // Double rAF ensures React has finished reconciling before we inject
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                var chatDiv = document.getElementById("chat-messages");
                if (!chatDiv) { _processing = false; return; }

                // Remove old injected elements
                var old = chatDiv.querySelectorAll(".aia-js-injected");
                old.forEach(function(el) { el.remove(); });

                // User bubble
                var bubble = document.createElement("div");
                bubble.textContent = question;
                bubble.className = "aia-js-injected";
                bubble.style.cssText =
                    "background-color:#c0392b;color:white;padding:10px 16px;" +
                    "border-radius:16px 16px 4px 16px;max-width:70%;" +
                    "margin-left:auto;margin-bottom:10px;font-size:0.9em;" +
                    "line-height:1.5;white-space:pre-wrap;word-break:break-word";
                chatDiv.appendChild(bubble);

                // Thinking block with spinner
                var thinking = document.createElement("div");
                thinking.className = "aia-js-injected";
                thinking.id = "aia-thinking-block";
                thinking.style.cssText =
                    "background-color:#f9fafb;border:1px solid #e5e7eb;" +
                    "border-radius:12px;padding:12px 14px;margin-bottom:6px;" +
                    "max-width:85%;font-size:0.8em;color:#6b7280";

                // Start with just the header + first step
                thinking.innerHTML =
                    '<div style="display:flex;align-items:center;margin-bottom:8px">' +
                      '<div style="width:14px;height:14px;border:2px solid #d97706;border-top-color:transparent;border-radius:50%;margin-right:8px;animation:aiaThinkSpin 0.8s linear infinite"></div>' +
                      '<span style="font-weight:600;color:#1f2937">Thinking</span>' +
                      '<span style="margin-left:2px;animation:aiaThinkPulse 1.5s ease-in-out infinite;color:#6b7280">...</span>' +
                    '</div>' +
                    '<div id="aia-think-steps">' + makeStepHtml(THINK_STEPS[0], true) + '</div>';

                chatDiv.appendChild(thinking);
                chatDiv.scrollTop = chatDiv.scrollHeight;

                // Disable input while processing
                var inp = document.getElementById("user-input");
                var btn = document.getElementById("send-btn");
                if (inp) { inp.disabled = true; inp.style.opacity = "0.5"; }
                if (btn) { btn.disabled = true; btn.style.opacity = "0.5"; }

                // Progressive step animation — add a new step every 2.5s
                var currentStep = 1;
                _stepInterval = setInterval(function() {
                    if (!_processing || currentStep >= THINK_STEPS.length) {
                        clearInterval(_stepInterval);
                        _stepInterval = null;
                        return;
                    }
                    var stepsDiv = document.getElementById("aia-think-steps");
                    if (stepsDiv) {
                        stepsDiv.innerHTML += makeStepHtml(THINK_STEPS[currentStep], true);
                        var cd = document.getElementById("chat-messages");
                        if (cd) cd.scrollTop = cd.scrollHeight;
                    }
                    currentStep++;
                }, 2500);
            });
        });
    }

    function cleanup() {
        if (_stepInterval) { clearInterval(_stepInterval); _stepInterval = null; }
        _processing = false;
        var chatDiv = document.getElementById("chat-messages");
        if (chatDiv) {
            var injected = chatDiv.querySelectorAll(".aia-js-injected");
            injected.forEach(function(el) { el.remove(); });
        }
        var inp = document.getElementById("user-input");
        var btn = document.getElementById("send-btn");
        if (inp) { inp.disabled = false; inp.style.opacity = "1"; inp.focus(); }
        if (btn) { btn.disabled = false; btn.style.opacity = "1"; }
    }

    function setupListeners() {
        var chatDiv = document.getElementById("chat-messages");
        var sendBtn = document.getElementById("send-btn");
        var userInput = document.getElementById("user-input");

        if (!chatDiv || !sendBtn || !userInput) {
            setTimeout(setupListeners, 500);
            return;
        }

        // Observe React updates to chat-messages — clean up injected elements + re-enable input
        var observer = new MutationObserver(function(mutations) {
            var hasReactAdds = false;
            mutations.forEach(function(m) {
                if (m.addedNodes.length > 0) {
                    m.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1 && !node.classList.contains("aia-js-injected")) {
                            hasReactAdds = true;
                        }
                    });
                }
            });

            if (hasReactAdds && _processing) {
                cleanup();
                requestAnimationFrame(function() {
                    var cd = document.getElementById("chat-messages");
                    if (cd) cd.scrollTop = cd.scrollHeight;
                });
            }
        });
        observer.observe(chatDiv, { childList: true });

        // Send button click — capture phase fires before Dash/React
        sendBtn.addEventListener("click", function() {
            var inp = document.getElementById("user-input");
            if (inp && inp.value && inp.value.trim()) {
                injectThinking(inp.value.trim());
            }
        }, true);

        // Enter key — capture phase
        userInput.addEventListener("keydown", function(e) {
            if (e.key === "Enter" && !e.shiftKey) {
                var inp = document.getElementById("user-input");
                if (inp && inp.value && inp.value.trim()) {
                    injectThinking(inp.value.trim());
                }
            }
        }, true);

        // Sample question clicks — event delegation on capture phase
        document.addEventListener("click", function(e) {
            var target = e.target;
            for (var i = 0; i < 5 && target; i++) {
                var rawId = target.getAttribute && target.getAttribute("id");
                if (rawId && rawId.indexOf("sample-q") >= 0) {
                    try {
                        var parsed = JSON.parse(rawId);
                        if (parsed && parsed.type === "sample-q" && typeof parsed.index === "number") {
                            injectThinking(SAMPLE_QS[parsed.index]);
                        }
                    } catch(ex) {}
                    return;
                }
                target = target.parentElement;
            }
        }, true);

        // New chat button — only reload if agent is processing (server blocked)
        var newChatBtn = document.getElementById("new-chat-btn");
        if (newChatBtn) {
            newChatBtn.addEventListener("click", function(e) {
                if (_processing) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                    window.location.reload();
                    return false;
                }
                // Let Dash callback handle it normally
            }, true);
        }

        console.log("[AIA] Thinking animation listeners attached");
    }

    // Start after DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function() { setTimeout(setupListeners, 500); });
    } else {
        setTimeout(setupListeners, 500);
    }
})();
