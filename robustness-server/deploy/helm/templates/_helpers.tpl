{{- define "robustness-server.name" -}}
{{- default "robustness-server" .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "robustness-server.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "robustness-server.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "robustness-server.labels" -}}
app.kubernetes.io/name: {{ include "robustness-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "robustness-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "robustness-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "robustness-server.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "robustness-server.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
