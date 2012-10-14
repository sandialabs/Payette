;;; A major mode for editing Payette input files
;;;
;;; Code:
(defvar payette-mode-hook nil)
(defvar payette-mode-map
  (let ((payette-mode-map (make-keymap)))
    (define-key payette-mode-map "\C-j" 'newline-and-indent)
    payette-mode-map)
  "Keymap for PAYETTE major mode")

(add-to-list 'auto-mode-alist '("\\.inp\\'" . payette-mode))



(defvar payette-begend "\\<\\(begin\\|end\\)\\>")

; to generate the payette-blocks execute:
; (regexp-opt '("simulation" "parameterization" "mathplot" "output" "boundary" "legs" "material" "extraction" "permutation" "optimization" "shearfit" "hydrofit" "preprocessing") 'words)
(defvar payette-blocks
  "\\<\\(preprocessing\\|boundary\\|extraction\\|hydrofit\\|legs\\|mat\\(?:erial\\|hplot\\)\\|o\\(?:ptimization\\|utput\\)\\|p\\(?:\\(?:arameteriz\\|ermut\\)ation\\)\\|s\\(?:hearfit\\|imulation\\)\\)\\>")

; to generate the payette-booleans execute:
; (regexp-opt '("nowriteprops" "use_table" "write_restart" "write_input" "write_curves") 'words)
(defvar payette-boolean
  "\\<\\(nowriteprops\\|use_table\\|write_\\(?:curves\\|\\(?:inpu\\|restar\\)t\\)\\)\\>")

; to generate the payette-directives execute:
; (regexp-opt '("error" "disp" "constitutive model" "using" "ampl" "screenout" "emit" "kappa" "stepstar" "estar" "sstar" "ratfac" "efstar" "dstar" "vstar" "tstar" "options" "maxiter" "method" "data file" "seed" "from" "obj_fn in" "obj_fn" "gold file" "tolerance" "matlabel" "input units" "output units" "density range" "temperature range" "surface increments" "path increments" "path isotherm" "path hugoniot") 'words)
(defvar payette-keywords
  "\\<\\(ampl\\|constitutive *model\\|d\\(?:ata *file\\|ensity *range\\|isp\\|star\\)\\|e\\(?:fstar\\|mit\\|\\(?:rro\\|sta\\)r\\)\\|gold *file\\|input *units\\|kappa\\|m\\(?:a\\(?:tlabel\\|xiter\\)\\|ethod\\)\\|o\\(?:bj_fn\\(?: *in\\)?\\|\\(?:ption\\|utput *unit\\)s\\)\\|path *\\(?:hugoniot\\|i\\(?:ncrements\\|sotherm\\)\\)\\|ratfac\\|s\\(?:creenout\\|eed\\|\\(?:tep\\)?star\\)\\|t\\(?:emperature *range\\|olerance\\|star\\)\\|vstar\\|surface *increments\\)\\>")

; to generate the payette-directives execute:
; (regexp-opt '("optimize" "permutate" "fix" "insert" "from" "from columns") 'words)
(defvar payette-directives
  "\\<\\(f\\(?:ix\\|rom\\(?: *columns\\)?\\)\\|insert\\|optimize\\|minimize\\|versus\\|permutate\\|using\\)\\>")

; to generate the payette-opts execute:
; (regexp-opt '("initial value" "bounds" "range" "sequence" "normal" "weibull" "+/-" "percentage" "uniform") 'words)
(defvar payette-opt-opts
  "\\<\\(\\+/-\\|bounds\\|initial value\\|normal\\|percentage\\|range\\|sequence\\|uniform\\|weibull\\)\\>")

; font-lock-(constant|keyword|type|function-name|builtin|variable-name)-face
; constant: sea-foam
; keyword: purple
; type: green
; function-name: blue
; builtin: pink
; variable-name: mustard
(setq payette-font-lock-keywords
      `(
	(,payette-keywords . font-lock-type-face)
	(,payette-begend . font-lock-keyword-face)
	(,payette-blocks . font-lock-function-name-face)
	(,payette-boolean . font-lock-builtin-face)
	(,payette-directives . font-lock-variable-name-face)
	(,payette-opt-opts . font-lock-constant-face)
	)
      )

(defvar payette-mode-syntax-table
  (let ((payette-mode-syntax-table (make-syntax-table)))

    ; This is added so entity names with underscores can be more easily parsed
    (modify-syntax-entry ?_ "w" payette-mode-syntax-table)

	; Comment styles are same as Python
    (modify-syntax-entry ?# "< b" payette-mode-syntax-table)
    (modify-syntax-entry ?\n "> b" payette-mode-syntax-table)
    payette-mode-syntax-table)
  "Syntax table for payette-mode")

(define-derived-mode payette-mode fundamental-mode
  "Payette mode"
  "Major mode for editing Payette input files"

  ;; code for syntax highlighting
  (setq font-lock-defaults '((payette-font-lock-keywords)))
  (set-syntax-table payette-mode-syntax-table)
  (setq-default font-lock-keywords-case-fold-search t)
)

(provide 'payette-mode)

;;; payette-mode.el ends here
